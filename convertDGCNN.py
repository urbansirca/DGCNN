import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import mne

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torcheeg.models import DGCNN

from torch_geometric.explain import Explainer, GNNExplainer
from torcheeg.datasets import SEEDDataset
from torcheeg import transforms
from contrastive_explainer import explain_class_contrast

# Frequency band names
FREQUENCY_BANDS = ['Delta (1-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-14 Hz)', 'Beta (14-31 Hz)', 'Gamma (31-49 Hz)']



# ============================================================================
# MODEL DEFINITIONS
# ============================================================================


class PyGGraphConvolution(MessagePassing):
    """
    Graph convolution using PyG's message passing for GNNExplainer compatibility.

    Original DGCNN order: out = (adj @ x) @ weight
    This must be preserved for equivalence.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        # IMPORTANT: Original DGCNN does (adj @ x) @ weight
        # Step 1: Propagate first (equivalent to adj @ x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        # Step 2: Then apply weight transformation
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class PyGChebynet(nn.Module):
    """
    Chebyshev graph convolution using PyG message passing.

    Original Chebynet generates Chebyshev polynomial bases:
    - T_0(L) = I (identity)
    - T_1(L) = L (normalized adjacency)
    - T_k(L) = 2*L*T_{k-1}(L) - T_{k-2}(L) (recursive)

    For GNNExplainer compatibility, we use edge_weight to control the adjacency.
    T_0 term: just x @ weight (identity, no graph propagation)
    T_1 term: (L @ x) @ weight (use normalized adjacency)
    """

    def __init__(self, in_channels: int, num_layers: int, out_channels: int):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList(
            [PyGGraphConvolution(in_channels, out_channels) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        self_loop_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges] - includes self-loops
            edge_weight: Normalized edge weights [num_edges] (from D^-0.5 @ A @ D^-0.5)
            num_nodes: Number of nodes
            self_loop_mask: Boolean mask indicating which edges are self-loops [num_edges]
        """
        result = None

        for k in range(self.num_layers):
            if k == 0:
                # T_0(L) * x = I * x
                # Identity: only use self-loops with weight 1
                if self_loop_mask is not None:
                    weights_k = torch.where(
                        self_loop_mask,
                        torch.ones_like(edge_weight),
                        torch.zeros_like(edge_weight),
                    )
                else:
                    # Fallback: use edge_weight as-is
                    weights_k = edge_weight
            else:
                # T_1(L) * x = L * x
                # Use the full normalized adjacency (including self-loops from normalization)
                weights_k = edge_weight

            conv_out = self.convs[k](x, edge_index, weights_k)

            if result is None:
                result = conv_out
            else:
                result = result + conv_out

        return F.relu(result)


class DGCNNForExplainerPyG(nn.Module):
    """
    DGCNN rewritten to use PyG message passing for GNNExplainer compatibility.
    """

    def __init__(self, trained_model):
        super().__init__()
        self.in_channels = trained_model.in_channels
        self.num_electrodes = trained_model.num_electrodes
        self.hid_channels = trained_model.hid_channels
        self.num_layers = trained_model.num_layers
        self.num_classes = trained_model.num_classes

        # Create new PyG-compatible layers
        self.layer1 = PyGChebynet(self.in_channels, self.num_layers, self.hid_channels)

        # Copy weights from trained model's Chebynet
        self._copy_chebynet_weights(trained_model.layer1)

        # Copy other layers directly
        self.BN1 = trained_model.BN1
        self.fc1 = trained_model.fc1
        self.fc2 = trained_model.fc2

        # Store learned adjacency
        self.register_buffer("learned_A", trained_model.A.data.clone())

        # Will be set when preparing edge_index
        self.register_buffer("self_loop_mask", None)

    def _copy_chebynet_weights(self, original_chebynet):
        """Copy weights from original Chebynet to PyG version."""
        for new_conv, old_conv in zip(self.layer1.convs, original_chebynet.gc1):
            new_conv.weight.data = old_conv.weight.data.clone()
            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data.clone()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass compatible with GNNExplainer.

        Args:
            x: Node features [num_nodes, in_channels] = [62, 5]
            edge_index: Edge indices [2, num_edges] - includes self-loops
            edge_weight: Edge weights [num_edges] - this is what GNNExplainer perturbs
        """
        # Handle input shape
        if x.dim() == 3:
            x = x.squeeze(0)

        # Apply batch norm
        x_bn = x.unsqueeze(0)
        x_bn = self.BN1(x_bn.transpose(1, 2)).transpose(1, 2)
        x = x_bn.squeeze(0)

        # Use provided edge_weight or compute from learned adjacency
        if edge_weight is None:
            edge_weight = self._get_edge_weights_from_adj(edge_index)

        # Create self-loop mask if not already set
        self_loop_mask = self._get_self_loop_mask(edge_index)

        # Graph convolution
        result = self.layer1(
            x, edge_index, edge_weight, self.num_electrodes, self_loop_mask
        )

        # Flatten and classify
        result = result.reshape(1, -1)
        result = F.relu(self.fc1(result))
        result = self.fc2(result)

        return result

    def _get_self_loop_mask(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Create a mask indicating which edges are self-loops."""
        return edge_index[0] == edge_index[1]

    def _get_edge_weights_from_adj(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Get edge weights from the learned adjacency matrix."""
        A = F.relu(self.learned_A)
        edge_weight = A[edge_index[0], edge_index[1]]
        return edge_weight

    def _get_normalized_edge_weights(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized edge weights matching original normalize_A function.

        Original normalize_A does:
            A = F.relu(A)
            d = torch.sum(A, 1)  # row sum of full matrix
            d = 1 / sqrt(d + 1e-10)
            D = diag(d)
            L = D @ A @ D  (symmetric normalization)

        IMPORTANT: edge_index is in transposed form for message passing:
        - edge_index[0] = source (j), edge_index[1] = target (i)
        - This represents L[i,j] for computing L @ x
        """
        A = F.relu(self.learned_A)

        # Compute degree from FULL adjacency matrix (row sums)
        deg = A.sum(dim=1)

        # Compute D^-0.5
        deg_inv_sqrt = (deg + 1e-10).pow(-0.5)

        # edge_index: [0]=source=j, [1]=target=i
        # We need weight L[i,j] = deg_inv_sqrt[i] * A[i,j] * deg_inv_sqrt[j]
        src, dst = edge_index  # src=j, dst=i
        # A[i,j] = A[dst, src]
        edge_weight = A[dst, src]
        # L[i,j] = D^-0.5[i] * A[i,j] * D^-0.5[j]
        normalized_weight = deg_inv_sqrt[dst] * edge_weight * deg_inv_sqrt[src]

        return normalized_weight

    def get_edge_index_and_attr(self, threshold: float = 0.0):
        """
        Convert learned adjacency to edge format.

        IMPORTANT: For matrix multiplication L @ x via message passing:
        - PyG aggregates messages at the TARGET node (edge_index[1])
        - For L[i,j] * x[j] to contribute to output[i], we need edge (j -> i)
        - So edge_index = [[j, ...], [i, ...]] with weight L[i,j]
        - This means we TRANSPOSE the usual edge representation

        For T_0 (identity), we need self-loops.
        For T_1 (adjacency), we need the actual adjacency edges.
        """
        A = F.relu(self.learned_A)

        # Get all edges above threshold
        if threshold > 0:
            mask = A > threshold
        else:
            mask = A > 1e-10

        # Remove diagonal - we'll add self-loops separately for T_0 handling
        diag_mask = torch.eye(self.num_electrodes, dtype=torch.bool, device=A.device)
        mask = mask & ~diag_mask

        # nonzero gives [row, col] = [i, j] where A[i,j] > threshold
        indices = torch.nonzero(mask, as_tuple=False).t().contiguous()
        # For L @ x, we need edge (j -> i), so SWAP to get [col, row] = [j, i]
        edge_index = torch.stack([indices[1], indices[0]], dim=0)
        # Weight is still A[i,j] = A[indices[0], indices[1]]
        edge_attr = A[indices[0], indices[1]]

        # Add self-loops (needed for T_0 identity operation)
        self_loops = torch.arange(self.num_electrodes, device=A.device)
        self_loop_index = torch.stack([self_loops, self_loops])
        # Self-loop weights set to 1 (identity for T_0)
        self_loop_attr = torch.ones(self.num_electrodes, device=A.device)

        # Concatenate: regular edges + self-loops
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)
        edge_attr = torch.cat([edge_attr, self_loop_attr])

        return edge_index, edge_attr


def prepare_for_explainer_pyg(trained_model, threshold: float = 0.0):
    """Prepare model for GNNExplainer using PyG-compatible version."""
    wrapper = DGCNNForExplainerPyG(trained_model)
    wrapper.eval()

    for param in wrapper.parameters():
        param.requires_grad = False

    edge_index, edge_attr = wrapper.get_edge_index_and_attr(threshold=threshold)

    return wrapper, edge_index, edge_attr


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

SEED_ELECTRODE_NAMES = [
    "FP1",
    "FPZ",
    "FP2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "OZ",
    "O2",
    "CB2",
]


def get_electrode_positions(num_electrodes: int = 62):
    """Generate 2D positions for electrodes in a head-like layout."""
    positions = {}

    if num_electrodes == 62:
        rows = {
            "frontal": ["FP1", "FPZ", "FP2"],
            "af": ["AF3", "AF4"],
            "f": ["F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"],
            "ft_fc": ["FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"],
            "temporal_central": ["T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"],
            "tp_cp": ["TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"],
            "parietal": ["P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"],
            "po": ["PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8"],
            "occipital": ["CB1", "O1", "OZ", "O2", "CB2"],
        }

        y_positions = {
            "frontal": 0.9,
            "af": 0.8,
            "f": 0.7,
            "ft_fc": 0.55,
            "temporal_central": 0.4,
            "tp_cp": 0.25,
            "parietal": 0.1,
            "po": -0.05,
            "occipital": -0.2,
        }

        for row_name, electrodes in rows.items():
            y = y_positions[row_name]
            n = len(electrodes)
            for i, elec in enumerate(electrodes):
                x = (i - (n - 1) / 2) * 0.2
                positions[elec] = (x, y)

    pos_array = np.zeros((num_electrodes, 2))
    for i, name in enumerate(SEED_ELECTRODE_NAMES[:num_electrodes]):
        if name in positions:
            pos_array[i] = positions[name]
        else:
            angle = 2 * np.pi * i / num_electrodes
            pos_array[i] = (np.cos(angle) * 0.8, np.sin(angle) * 0.8)

    return pos_array


def visualize_node_importance(
    node_mask: torch.Tensor,
    electrode_names: list = None,
    title: str = "Node Importance",
    save_path: str = None,
):
    """Visualize node (electrode) importance on a head layout."""
    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES

    num_electrodes = len(node_mask)
    positions = get_electrode_positions(num_electrodes)

    if node_mask.dim() > 1:
        importance = node_mask.mean(dim=-1).cpu().numpy()
    else:
        importance = node_mask.cpu().numpy()

    importance = (importance - importance.min()) / (
        importance.max() - importance.min() + 1e-10
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    circle = plt.Circle((0, 0.35), 0.65, fill=False, color="black", linewidth=2)
    ax.add_patch(circle)
    ax.plot([0, 0.1, 0], [1.0, 1.1, 1.0], "k-", linewidth=2)
    ax.plot([-0.65, -0.7, -0.65], [0.3, 0.35, 0.4], "k-", linewidth=2)
    ax.plot([0.65, 0.7, 0.65], [0.3, 0.35, 0.4], "k-", linewidth=2)

    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=importance,
        cmap="Reds",
        s=500,
        edgecolors="black",
        linewidths=1,
        vmin=0,
        vmax=1,
    )

    for i, (x, y) in enumerate(positions):
        ax.annotate(
            electrode_names[i] if i < len(electrode_names) else str(i),
            (x, y),
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
        )

    plt.colorbar(scatter, ax=ax, label="Importance")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fig


def visualize_edge_importance(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    num_electrodes: int = 62,
    electrode_names: list = None,
    top_k: int = 50,
    title: str = "Edge Importance",
    save_path: str = None,
):
    """Visualize the most important edges (connections) between electrodes."""
    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES

    positions = get_electrode_positions(num_electrodes)

    edge_importance = edge_mask.cpu().numpy()
    edge_idx_np = edge_index.cpu().numpy()

    # Filter out self-loops for visualization
    non_self_loop = edge_idx_np[0] != edge_idx_np[1]
    edge_importance_filtered = edge_importance[non_self_loop]
    edge_idx_filtered = edge_idx_np[:, non_self_loop]

    # Get top-k edges
    top_indices = np.argsort(edge_importance_filtered)[-top_k:]

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    circle = plt.Circle((0, 0.35), 0.65, fill=False, color="black", linewidth=2)
    ax.add_patch(circle)
    ax.plot([0, 0.1, 0], [1.0, 1.1, 1.0], "k-", linewidth=2)
    ax.plot([-0.65, -0.7, -0.65], [0.3, 0.35, 0.4], "k-", linewidth=2)
    ax.plot([0.65, 0.7, 0.65], [0.3, 0.35, 0.4], "k-", linewidth=2)

    top_importance = edge_importance_filtered[top_indices]
    norm_importance = (top_importance - top_importance.min()) / (
        top_importance.max() - top_importance.min() + 1e-10
    )

    for idx, imp in zip(top_indices, norm_importance):
        src, dst = edge_idx_filtered[0, idx], edge_idx_filtered[1, idx]
        if src < num_electrodes and dst < num_electrodes:
            x = [positions[src, 0], positions[dst, 0]]
            y = [positions[src, 1], positions[dst, 1]]
            ax.plot(
                x,
                y,
                color=plt.cm.Reds(imp),
                alpha=0.3 + 0.7 * imp,
                linewidth=1 + 3 * imp,
            )

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c="lightblue",
        s=300,
        edgecolors="black",
        linewidths=1,
        zorder=10,
    )

    for i, (x, y) in enumerate(positions):
        ax.annotate(
            electrode_names[i] if i < len(electrode_names) else str(i),
            (x, y),
            ha="center",
            va="center",
            fontsize=5,
            fontweight="bold",
            zorder=11,
        )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title} (Top {top_k} edges)", fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fig


def visualize_learned_adjacency(
    model,
    threshold: float = 0.1,
    title: str = "Learned Adjacency Matrix",
    save_path: str = None,
):
    """Visualize the learned adjacency matrix as a heatmap."""
    A = F.relu(model.A.data).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    im1 = axes[0].imshow(A, cmap="hot", aspect="auto")
    axes[0].set_title("Learned Adjacency Matrix", fontsize=12)
    axes[0].set_xlabel("Electrode")
    axes[0].set_ylabel("Electrode")
    plt.colorbar(im1, ax=axes[0])

    A_thresh = A.copy()
    A_thresh[A_thresh < threshold] = 0
    im2 = axes[1].imshow(A_thresh, cmap="hot", aspect="auto")
    axes[1].set_title(f"Thresholded (>{threshold})", fontsize=12)
    axes[1].set_xlabel("Electrode")
    axes[1].set_ylabel("Electrode")
    plt.colorbar(im2, ax=axes[1])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fig


def get_aggregated_explanations(
    explainer,
    data_loader,
    edge_index,
    edge_weight,
    num_samples_per_class: int = 50,
    num_classes: int = 3,
):
    """Aggregate explanations over multiple samples per class."""

    class_node_masks = {i: [] for i in range(num_classes)}
    class_edge_masks = {i: [] for i in range(num_classes)}
    sample_counts = {i: 0 for i in range(num_classes)}

    print(f"Collecting {num_samples_per_class} samples per class...")

    for x, y in data_loader:
        for i in range(len(y)):
            label = y[i].item()

            if sample_counts[label] >= num_samples_per_class:
                continue

            # Generate explanation for this sample
            sample = x[i].squeeze()
            explanation = explainer(
                x=sample,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

            class_node_masks[label].append(explanation.node_mask.detach().cpu())
            class_edge_masks[label].append(explanation.edge_mask.detach().cpu())
            sample_counts[label] += 1

            total = sum(sample_counts.values())
            print(f"  Processed {total}/{num_samples_per_class * num_classes} samples", end="\r")

        if all(c >= num_samples_per_class for c in sample_counts.values()):
            break

    print("\n\nAggregating explanations...")

    # Compute statistics for each class
    aggregated = {}
    for label in range(num_classes):
        if len(class_node_masks[label]) == 0:
            continue

        node_masks = torch.stack(class_node_masks[label])  # [n_samples, 62, 5]
        edge_masks = torch.stack(class_edge_masks[label])  # [n_samples, 1378]

        aggregated[label] = {
            'node_mask_mean': node_masks.mean(dim=0),
            'node_mask_std': node_masks.std(dim=0),
            'edge_mask_mean': edge_masks.mean(dim=0),
            'edge_mask_std': edge_masks.std(dim=0),
            'num_samples': len(class_node_masks[label]),
        }

    return aggregated


def visualize_edge_importance_circular(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    num_electrodes: int = 62,
    electrode_names: list = None,
    n_lines: int = 50,
    title: str = "Edge Importance",
    save_path: str = None,
):
    """Visualize edge importance using MNE's circular connectivity plot."""
    from mne.viz import circular_layout
    from mne_connectivity.viz import plot_connectivity_circle

    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES[:num_electrodes]

    edge_idx_np = edge_index.cpu().numpy()
    edge_importance = edge_mask.cpu().numpy()

    # Build connectivity matrix from edge_index and edge_mask
    conn_matrix = np.zeros((num_electrodes, num_electrodes))
    for i in range(edge_idx_np.shape[1]):
        src, dst = edge_idx_np[0, i], edge_idx_np[1, i]
        if src != dst:  # Skip self-loops
            conn_matrix[src, dst] = edge_importance[i]

    # Make symmetric for visualization
    conn_matrix = (conn_matrix + conn_matrix.T) / 2

    # Group electrodes by hemisphere
    # Left hemisphere electrodes
    lh_electrodes = [
        name for name in electrode_names if name.endswith(("1", "3", "5", "7"))
    ]

    # Midline electrodes
    mid_electrodes = [name for name in electrode_names if name.endswith("Z")]

    # Right hemisphere electrodes
    rh_electrodes = [
        name for name in electrode_names if name.endswith(("2", "4", "6", "8"))
    ]

    # Create node order: left -> midline -> right
    node_order = lh_electrodes + mid_electrodes + rh_electrodes

    # Create circular layout with group boundaries
    node_angles = circular_layout(
        electrode_names,
        node_order,
        start_pos=90,
        group_boundaries=[
            0,
            len(lh_electrodes),
            len(lh_electrodes) + len(mid_electrodes),
        ],
    )

    # Assign colors by hemisphere
    node_colors = []
    for name in electrode_names:
        if name in lh_electrodes:
            node_colors.append("#4A90D9")  # Blue - left hemisphere
        elif name in mid_electrodes:
            node_colors.append("#2ECC71")  # Green - midline
        else:
            node_colors.append("#E74C3C")  # Red - right hemisphere

    fig, ax = plt.subplots(
        figsize=(12, 12), facecolor="black", subplot_kw=dict(polar=True)
    )

    plot_connectivity_circle(
        conn_matrix,
        electrode_names,
        n_lines=n_lines,
        node_angles=node_angles,
        node_colors=node_colors,
        title=title,
        ax=ax,
        colormap="hot",
        fontsize_names=7,
        fontsize_title=14,
        padding=2.0,
        show=False,
        vmin=0,
        vmax=conn_matrix.max(),
    )

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.show()

    return fig


def visualize_edge_importance_circular_subplots(
    edge_index: torch.Tensor,
    edge_masks: dict,
    class_names: list,
    num_electrodes: int = 62,
    electrode_names: list = None,
    n_lines: int = 50,
    title: str = "Edge Importance by Class",
    save_path: str = None,
    ncols: int = None,
    normalize_per_plot: bool = False,
):
    """
    Visualize edge importance for multiple classes as subplots.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_masks: Dictionary mapping class_idx/key -> edge_mask tensor
        class_names: List of class names (in order of edge_masks keys)
        num_electrodes: Number of electrodes
        electrode_names: List of electrode names
        n_lines: Number of top connections to display
        title: Overall figure title
        save_path: Path to save the figure
        ncols: Number of columns (auto-determined if None)
        normalize_per_plot: If True, normalize each plot independently (useful for contrastive)
    """
    from mne.viz import circular_layout
    from mne_connectivity.viz import plot_connectivity_circle

    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES[:num_electrodes]

    edge_idx_np = edge_index.cpu().numpy()

    # Group electrodes by hemisphere
    lh_electrodes = [
        name for name in electrode_names if name.endswith(("1", "3", "5", "7"))
    ]
    mid_electrodes = [name for name in electrode_names if name.endswith("Z")]
    rh_electrodes = [
        name for name in electrode_names if name.endswith(("2", "4", "6", "8"))
    ]

    node_order = lh_electrodes + mid_electrodes + rh_electrodes

    node_angles = circular_layout(
        electrode_names,
        node_order,
        start_pos=90,
        group_boundaries=[
            0,
            len(lh_electrodes),
            len(lh_electrodes) + len(mid_electrodes),
        ],
    )

    node_colors = []
    for name in electrode_names:
        if name in lh_electrodes:
            node_colors.append("#4A90D9")
        elif name in mid_electrodes:
            node_colors.append("#2ECC71")
        else:
            node_colors.append("#E74C3C")

    num_plots = len(edge_masks)

    # Determine grid layout
    if ncols is None:
        if num_plots <= 3:
            ncols = num_plots
        else:
            ncols = 3
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 5 * nrows),
        facecolor="black",
        subplot_kw=dict(polar=True),
    )

    # Flatten axes for easy iteration
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Build connectivity matrices
    global_max = 0
    conn_matrices = {}
    for key, edge_mask in edge_masks.items():
        edge_importance = edge_mask.cpu().numpy()
        conn_matrix = np.zeros((num_electrodes, num_electrodes))
        for i in range(edge_idx_np.shape[1]):
            src, dst = edge_idx_np[0, i], edge_idx_np[1, i]
            if src != dst:
                conn_matrix[src, dst] = edge_importance[i]
        conn_matrix = (conn_matrix + conn_matrix.T) / 2
        conn_matrices[key] = conn_matrix
        global_max = max(global_max, conn_matrix.max())

    for ax_idx, (key, conn_matrix) in enumerate(conn_matrices.items()):
        # Get label from class_names list
        label = class_names[ax_idx] if ax_idx < len(class_names) else str(key)

        # Use per-plot or global normalization
        vmax = conn_matrix.max() if normalize_per_plot else global_max

        plot_connectivity_circle(
            conn_matrix,
            electrode_names,
            n_lines=n_lines,
            node_angles=node_angles,
            node_colors=node_colors,
            title=label,
            ax=axes[ax_idx],
            colormap="hot",
            fontsize_names=5,
            fontsize_title=10,
            padding=2.0,
            show=False,
            vmin=0,
            vmax=vmax,
        )

    # Hide unused axes
    for ax_idx in range(num_plots, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(title, fontsize=14, color="white", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.show()

    return fig


def visualize_node_importance_subplots(
    node_masks: dict,
    class_names: list,
    electrode_names: list = None,
    title: str = "Node Importance by Class",
    save_path: str = None,
    ncols: int = None,
    normalize_per_plot: bool = False,
):
    """
    Visualize node importance for multiple classes as subplots.

    Args:
        node_masks: Dictionary mapping class_idx/key -> node_mask tensor
        class_names: List of class names (in order of node_masks keys)
        electrode_names: List of electrode names
        title: Overall figure title
        save_path: Path to save the figure
        ncols: Number of columns (auto-determined if None)
        normalize_per_plot: If True, normalize each plot independently (useful for contrastive)
    """
    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES

    num_plots = len(node_masks)

    # Determine grid layout
    if ncols is None:
        if num_plots <= 3:
            ncols = num_plots
        else:
            ncols = 3
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # Flatten axes for easy iteration
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Get first mask to determine num_electrodes
    first_mask = list(node_masks.values())[0]
    num_electrodes = len(first_mask) if first_mask.dim() == 1 else first_mask.shape[0]
    positions = get_electrode_positions(num_electrodes)

    # Normalize all masks together for consistent color scaling
    all_importances = []
    for node_mask in node_masks.values():
        if node_mask.dim() > 1:
            importance = node_mask.mean(dim=-1).cpu().numpy()
        else:
            importance = node_mask.cpu().numpy()
        all_importances.append(importance)

    all_importances = np.array(all_importances)
    global_min = all_importances.min()
    global_max = all_importances.max()

    for ax_idx, (key, node_mask) in enumerate(node_masks.items()):
        ax = axes[ax_idx]

        if node_mask.dim() > 1:
            importance = node_mask.mean(dim=-1).cpu().numpy()
        else:
            importance = node_mask.cpu().numpy()

        # Normalize using global or per-plot min/max
        if normalize_per_plot:
            local_min, local_max = importance.min(), importance.max()
            importance_norm = (importance - local_min) / (local_max - local_min + 1e-10)
        else:
            importance_norm = (importance - global_min) / (global_max - global_min + 1e-10)

        # Draw head outline
        circle = plt.Circle((0, 0.35), 0.65, fill=False, color="black", linewidth=2)
        ax.add_patch(circle)
        ax.plot([0, 0.1, 0], [1.0, 1.1, 1.0], "k-", linewidth=2)
        ax.plot([-0.65, -0.7, -0.65], [0.3, 0.35, 0.4], "k-", linewidth=2)
        ax.plot([0.65, 0.7, 0.65], [0.3, 0.35, 0.4], "k-", linewidth=2)

        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=importance_norm,
            cmap="Reds",
            s=200,
            edgecolors="black",
            linewidths=1,
            vmin=0,
            vmax=1,
        )

        for i, (x, y) in enumerate(positions):
            ax.annotate(
                electrode_names[i] if i < len(electrode_names) else str(i),
                (x, y),
                ha="center",
                va="center",
                fontsize=4,
                fontweight="bold",
            )

        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

        # Get label from class_names list
        label = class_names[ax_idx] if ax_idx < len(class_names) else str(key)
        ax.set_title(label, fontsize=10)

    # Hide unused axes
    for ax_idx in range(num_plots, len(axes)):
        axes[ax_idx].set_visible(False)

    # Add a shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Importance")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fig


# ============================================================================
# MNE ELECTRODE POSITIONS & TOPOGRAPHIC VISUALIZATION
# ============================================================================

def get_mne_montage():
    """
    Create an MNE montage for the 62-channel SEED electrode layout.
    Maps SEED electrode names to standard 10-20 positions.
    """
    standard_montage = mne.channels.make_standard_montage('standard_1020')
    standard_pos = standard_montage.get_positions()['ch_pos']

    # Map SEED names to standard names
    name_mapping = {
        'FP1': 'Fp1', 'FPZ': 'Fpz', 'FP2': 'Fp2',
        'AF3': 'AF3', 'AF4': 'AF4',
        'F7': 'F7', 'F5': 'F5', 'F3': 'F3', 'F1': 'F1', 'FZ': 'Fz',
        'F2': 'F2', 'F4': 'F4', 'F6': 'F6', 'F8': 'F8',
        'FT7': 'FT7', 'FC5': 'FC5', 'FC3': 'FC3', 'FC1': 'FC1', 'FCZ': 'FCz',
        'FC2': 'FC2', 'FC4': 'FC4', 'FC6': 'FC6', 'FT8': 'FT8',
        'T7': 'T7', 'C5': 'C5', 'C3': 'C3', 'C1': 'C1', 'CZ': 'Cz',
        'C2': 'C2', 'C4': 'C4', 'C6': 'C6', 'T8': 'T8',
        'TP7': 'TP7', 'CP5': 'CP5', 'CP3': 'CP3', 'CP1': 'CP1', 'CPZ': 'CPz',
        'CP2': 'CP2', 'CP4': 'CP4', 'CP6': 'CP6', 'TP8': 'TP8',
        'P7': 'P7', 'P5': 'P5', 'P3': 'P3', 'P1': 'P1', 'PZ': 'Pz',
        'P2': 'P2', 'P4': 'P4', 'P6': 'P6', 'P8': 'P8',
        'PO7': 'PO7', 'PO5': 'PO5', 'PO3': 'PO3', 'POZ': 'POz',
        'PO4': 'PO4', 'PO6': 'PO6', 'PO8': 'PO8',
        'O1': 'O1', 'OZ': 'Oz', 'O2': 'O2',
    }

    positions = {}
    for seed_name in SEED_ELECTRODE_NAMES:
        std_name = name_mapping.get(seed_name, seed_name)
        if std_name in standard_pos:
            positions[seed_name] = standard_pos[std_name]
        elif seed_name == 'CB1':
            # Place CB1 (cerebellum left) below and left of O1
            o1_pos = standard_pos.get('O1', np.array([-0.03, -0.1, 0]))
            positions[seed_name] = o1_pos + np.array([-0.02, -0.03, -0.02])
        elif seed_name == 'CB2':
            # Place CB2 (cerebellum right) below and right of O2
            o2_pos = standard_pos.get('O2', np.array([0.03, -0.1, 0]))
            positions[seed_name] = o2_pos + np.array([0.02, -0.03, -0.02])
        else:
            positions[seed_name] = np.array([0, 0, 0])

    montage = mne.channels.make_dig_montage(ch_pos=positions, coord_frame='head')
    return montage


def get_mne_info():
    """Create MNE Info object for the 62-channel SEED setup."""
    info = mne.create_info(ch_names=SEED_ELECTRODE_NAMES, sfreq=200, ch_types='eeg')
    montage = get_mne_montage()
    info.set_montage(montage)
    return info


def plot_topomap_node_importance(
    node_importance,
    title="Node Importance",
    save_path=None,
    ax=None,
    cmap='Reds',
    vmin=None,
    vmax=None,
):
    """
    Plot node importance on an anatomically correct head topography using MNE.

    Args:
        node_importance: Array of shape [62] or [62, 5] (electrodes x features)
        title: Plot title
        save_path: Path to save figure
        ax: Matplotlib axis (creates new figure if None)
        cmap: Colormap
        vmin, vmax: Color scale limits
    """
    info = get_mne_info()

    if isinstance(node_importance, torch.Tensor):
        node_importance = node_importance.cpu().numpy()

    if node_importance.ndim > 1:
        node_importance = node_importance.mean(axis=-1)

    if vmin is None:
        vmin = node_importance.min()
    if vmax is None:
        vmax = node_importance.max()

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_fig = True

    im, _ = mne.viz.plot_topomap(
        node_importance,
        info,
        axes=ax,
        cmap=cmap,
        vlim=(vmin, vmax),
        show=False,
        contours=6,
        sensors=True,
    )

    ax.set_title(title, fontsize=12)

    if created_fig:
        plt.colorbar(im, ax=ax, label='Importance', shrink=0.8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved topomap to {save_path}")
        plt.close(fig)
        return fig

    return im


def plot_topomap_node_importance_subplots(
    node_masks: dict,
    class_names: list,
    title: str = "Node Importance - MNE Topomap",
    save_path: str = None,
    normalize_global: bool = True,
):
    """
    Plot node importance topomaps for multiple classes as subplots.
    """
    num_plots = len(node_masks)
    ncols = min(num_plots, 3)
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Compute global min/max
    all_values = []
    processed_masks = {}
    for key, mask in node_masks.items():
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.ndim > 1:
            mask = mask.mean(axis=-1)
        processed_masks[key] = mask
        all_values.extend(mask.flatten())

    if normalize_global:
        vmin, vmax = min(all_values), max(all_values)
    else:
        vmin, vmax = None, None

    info = get_mne_info()

    for idx, (key, importance) in enumerate(processed_masks.items()):
        ax = axes[idx]
        label = class_names[idx] if idx < len(class_names) else str(key)

        im, _ = mne.viz.plot_topomap(
            importance,
            info,
            axes=ax,
            cmap='Reds',
            vlim=(vmin, vmax) if normalize_global else (importance.min(), importance.max()),
            show=False,
            contours=6,
            sensors=True,
        )
        ax.set_title(label, fontsize=12)

    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label='Importance')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved topomap subplots to {save_path}")
    plt.close(fig)

    return fig


def plot_topomap_edge_importance_subplots(
    edge_masks: dict,
    class_names: list,
    edge_index: torch.Tensor,
    num_electrodes: int = 62,
    title: str = "Edge Importance - MNE Topomap",
    save_path: str = None,
    top_k: int = 30,
    normalize_global: bool = True,
):
    """
    Plot edge importance as connections on anatomically correct head topography.
    """
    num_plots = len(edge_masks)
    ncols = min(num_plots, 3)
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    info = get_mne_info()
    pos = np.array([info['chs'][i]['loc'][:2] for i in range(len(info['chs']))])

    # Convert all masks to adjacency matrices
    edge_idx_np = edge_index.cpu().numpy()
    adj_matrices = {}
    for key, mask in edge_masks.items():
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.ndim == 1:
            adj = np.zeros((num_electrodes, num_electrodes))
            for i in range(edge_idx_np.shape[1]):
                src, dst = edge_idx_np[0, i], edge_idx_np[1, i]
                if src != dst:
                    adj[src, dst] = mask[i]
            mask = adj
        mask = (mask + mask.T) / 2
        np.fill_diagonal(mask, 0)
        adj_matrices[key] = mask

    if normalize_global:
        global_max = max(m.max() for m in adj_matrices.values())

    for idx, (key, adj_matrix) in enumerate(adj_matrices.items()):
        ax = axes[idx]
        label = class_names[idx] if idx < len(class_names) else str(key)

        # Draw head outline
        head_radius = 0.1
        circle = plt.Circle((0, 0), head_radius, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        ax.plot([0, 0.02, 0], [head_radius, head_radius * 1.1, head_radius], 'k-', linewidth=2)
        ax.plot([-head_radius, -head_radius * 1.05, -head_radius], [-0.01, 0, 0.01], 'k-', linewidth=2)
        ax.plot([head_radius, head_radius * 1.05, head_radius], [-0.01, 0, 0.01], 'k-', linewidth=2)

        # Get top-k edges
        flat_mask = adj_matrix.flatten()
        top_indices = np.argsort(flat_mask)[-top_k * 2:]
        top_edges = np.unravel_index(top_indices, adj_matrix.shape)

        norm_max = global_max if normalize_global else adj_matrix.max()

        # Draw edges
        drawn_edges = set()
        for i, (src, dst) in enumerate(zip(top_edges[0], top_edges[1])):
            if src == dst:
                continue
            edge_key = (min(src, dst), max(src, dst))
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)

            x = [pos[src, 0], pos[dst, 0]]
            y = [pos[src, 1], pos[dst, 1]]
            importance = adj_matrix[src, dst] / norm_max if norm_max > 0 else 0

            ax.plot(x, y, color=plt.cm.Reds(importance),
                    alpha=0.3 + 0.7 * importance,
                    linewidth=1 + 3 * importance,
                    zorder=1)

        ax.scatter(pos[:, 0], pos[:, 1], c='lightblue', s=60,
                   edgecolors='black', linewidths=0.5, zorder=10)

        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(-0.15, 0.15)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(label, fontsize=11)

    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved edge topomap subplots to {save_path}")
    plt.close(fig)

    return fig


def plot_frequency_band_topomaps(
    node_mask,
    title: str = "Frequency Band Importance",
    save_path: str = None,
):
    """
    Plot separate topomaps for each frequency band.

    Args:
        node_mask: Array of shape [62, 5] (electrodes x frequency bands)
    """
    if isinstance(node_mask, torch.Tensor):
        node_mask = node_mask.cpu().numpy()

    if node_mask.ndim == 1:
        print("Warning: node_mask is 1D, cannot plot per-band topomaps")
        return None

    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    info = get_mne_info()

    vmin, vmax = node_mask.min(), node_mask.max()

    for idx, (ax, band_name) in enumerate(zip(axes, band_names)):
        band_importance = node_mask[:, idx]

        im, _ = mne.viz.plot_topomap(
            band_importance,
            info,
            axes=ax,
            cmap='Reds',
            vlim=(vmin, vmax),
            show=False,
            contours=4,
            sensors=True,
        )
        ax.set_title(f'{band_name}\n({FREQUENCY_BANDS[idx].split()[1]})', fontsize=10)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    plt.colorbar(im, cax=cbar_ax, label='Importance')

    fig.suptitle(title, fontsize=14, y=1.05)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved frequency band topomaps to {save_path}")
    plt.close(fig)

    return fig


# ============================================================================
# VALIDATION METRICS
# ============================================================================

def compute_fidelity_plus(
    model,
    samples: list,
    edge_masks: list,
    edge_index: torch.Tensor,
    top_k_percentile: int = 20,
    device: str = 'cpu',
):
    """
    Fidelity+ (Sufficiency): Remove top-k% important edges and measure accuracy DROP.

    High Fidelity+ means: removing important edges significantly hurts accuracy.
    This validates that the explanation identifies truly important edges.

    Returns:
        dict with original accuracy, masked accuracy, and fidelity+ score
    """
    model.eval()

    original_correct = 0
    masked_correct = 0
    total = len(samples)

    for i, (x, y) in enumerate(samples):
        if isinstance(x, torch.Tensor):
            x = x.squeeze()
        else:
            x = torch.tensor(x).squeeze()

        edge_mask = edge_masks[i]
        if isinstance(edge_mask, torch.Tensor):
            edge_mask = edge_mask.cpu().numpy()

        # Original prediction
        with torch.no_grad():
            edge_weight = model._get_normalized_edge_weights(edge_index)
            pred_orig = model(x, edge_index, edge_weight).argmax(dim=1).item()
            if pred_orig == y:
                original_correct += 1

        # Find threshold for top-k% edges
        threshold = np.percentile(edge_mask.flatten(), 100 - top_k_percentile)

        # Create masked edge weights (remove important edges)
        # Put mask on same device as edge_weight
        mask_tensor = torch.tensor(edge_mask >= threshold, device=edge_weight.device)
        masked_edge_weight = edge_weight.clone()
        masked_edge_weight[mask_tensor] = 0

        # Masked prediction
        with torch.no_grad():
            pred_masked = model(x, edge_index, masked_edge_weight).argmax(dim=1).item()
            if pred_masked == y:
                masked_correct += 1

    original_acc = original_correct / total
    masked_acc = masked_correct / total
    fidelity_plus = original_acc - masked_acc

    return {
        'original_accuracy': original_acc,
        'masked_accuracy': masked_acc,
        'fidelity_plus': fidelity_plus,
        'top_k_percentile': top_k_percentile
    }


def compute_fidelity_minus(
    model,
    samples: list,
    edge_masks: list,
    edge_index: torch.Tensor,
    top_k_percentile: int = 20,
    device: str = 'cpu',
):
    """
    Fidelity- (Comprehensiveness): Keep ONLY top-k% important edges and measure accuracy.

    High Fidelity- means: keeping only important edges maintains accuracy.
    This validates that the explanation captures sufficient information.
    """
    model.eval()

    original_correct = 0
    sparse_correct = 0
    total = len(samples)

    for i, (x, y) in enumerate(samples):
        if isinstance(x, torch.Tensor):
            x = x.squeeze()
        else:
            x = torch.tensor(x).squeeze()

        edge_mask = edge_masks[i]
        if isinstance(edge_mask, torch.Tensor):
            edge_mask = edge_mask.cpu().numpy()

        # Original prediction
        with torch.no_grad():
            edge_weight = model._get_normalized_edge_weights(edge_index)
            pred_orig = model(x, edge_index, edge_weight).argmax(dim=1).item()
            if pred_orig == y:
                original_correct += 1

        # Find threshold for top-k% edges
        threshold = np.percentile(edge_mask.flatten(), 100 - top_k_percentile)

        # Create sparse edge weights (keep only important edges)
        # Put mask on same device as edge_weight
        mask_tensor = torch.tensor(edge_mask >= threshold, device=edge_weight.device)
        sparse_edge_weight = torch.zeros_like(edge_weight)
        sparse_edge_weight[mask_tensor] = edge_weight[mask_tensor]

        # Sparse prediction
        with torch.no_grad():
            pred_sparse = model(x, edge_index, sparse_edge_weight).argmax(dim=1).item()
            if pred_sparse == y:
                sparse_correct += 1

    original_acc = original_correct / total
    sparse_acc = sparse_correct / total
    fidelity_minus = sparse_acc

    return {
        'original_accuracy': original_acc,
        'sparse_accuracy': sparse_acc,
        'fidelity_minus': fidelity_minus,
        'top_k_percentile': top_k_percentile
    }


def compute_sparsity_curve(
    model,
    samples: list,
    edge_masks: list,
    edge_index: torch.Tensor,
    percentiles: list = None,
    device: str = 'cpu',
):
    """
    Compute accuracy vs sparsity curve.

    Shows how accuracy changes as we keep fewer edges (only the most important ones).
    A good explanation should maintain high accuracy with few edges.
    """
    if percentiles is None:
        percentiles = [5, 10, 20, 30, 50, 70, 100]

    results = {'percentiles': percentiles, 'accuracies': []}

    for pct in percentiles:
        fidelity_result = compute_fidelity_minus(model, samples, edge_masks, edge_index,
                                                  top_k_percentile=pct, device=device)
        results['accuracies'].append(fidelity_result['sparse_accuracy'])

    return results


def compute_stability(edge_masks_list: list):
    """
    Compute stability/consistency of explanations across multiple samples.

    High stability means the explainer gives consistent results for similar inputs.
    Uses Intersection over Union (IoU) of top-k edges.
    """
    n_samples = len(edge_masks_list)
    if n_samples < 2:
        return {'mean_iou': 1.0, 'std_iou': 0.0, 'n_comparisons': 0}

    ious = []
    top_k_percent = 10

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            mask_i = edge_masks_list[i]
            mask_j = edge_masks_list[j]

            if isinstance(mask_i, torch.Tensor):
                mask_i = mask_i.cpu().numpy()
            if isinstance(mask_j, torch.Tensor):
                mask_j = mask_j.cpu().numpy()

            mask_i = mask_i.flatten()
            mask_j = mask_j.flatten()

            threshold_i = np.percentile(mask_i, 100 - top_k_percent)
            threshold_j = np.percentile(mask_j, 100 - top_k_percent)

            top_i = set(np.where(mask_i >= threshold_i)[0])
            top_j = set(np.where(mask_j >= threshold_j)[0])

            intersection = len(top_i & top_j)
            union = len(top_i | top_j)
            iou = intersection / union if union > 0 else 0
            ious.append(iou)

    return {
        'mean_iou': np.mean(ious),
        'std_iou': np.std(ious),
        'n_comparisons': len(ious)
    }


def compute_all_validation_metrics(
    model,
    samples: list,
    edge_masks: list,
    edge_index: torch.Tensor,
    device: str = 'cpu',
):
    """
    Compute all validation metrics for explanation results.
    """
    print("\nComputing validation metrics...")

    print("  Computing Fidelity+ (sufficiency)...")
    fidelity_plus = compute_fidelity_plus(model, samples, edge_masks, edge_index,
                                           top_k_percentile=20, device=device)

    print("  Computing Fidelity- (comprehensiveness)...")
    fidelity_minus = compute_fidelity_minus(model, samples, edge_masks, edge_index,
                                             top_k_percentile=20, device=device)

    print("  Computing sparsity curve...")
    sparsity = compute_sparsity_curve(model, samples, edge_masks, edge_index, device=device)

    print("  Computing stability...")
    stability = compute_stability(edge_masks)

    return {
        'fidelity_plus': fidelity_plus,
        'fidelity_minus': fidelity_minus,
        'sparsity_curve': sparsity,
        'stability': stability
    }


def print_validation_metrics(metrics: dict, name: str):
    """Print validation metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"VALIDATION METRICS - {name}")
    print(f"{'='*60}")

    fp = metrics['fidelity_plus']
    print(f"\nFidelity+ (Sufficiency) - Removing top {fp['top_k_percentile']}% edges:")
    print(f"  Original accuracy: {fp['original_accuracy']:.4f}")
    print(f"  After removal:     {fp['masked_accuracy']:.4f}")
    print(f"  Fidelity+ score:   {fp['fidelity_plus']:.4f} (higher = better explanation)")

    fm = metrics['fidelity_minus']
    print(f"\nFidelity- (Comprehensiveness) - Keeping only top {fm['top_k_percentile']}% edges:")
    print(f"  Original accuracy: {fm['original_accuracy']:.4f}")
    print(f"  Sparse accuracy:   {fm['sparse_accuracy']:.4f}")
    print(f"  Fidelity- score:   {fm['fidelity_minus']:.4f} (higher = better explanation)")

    stab = metrics['stability']
    print(f"\nStability (IoU of top edges across samples):")
    print(f"  Mean IoU:          {stab['mean_iou']:.4f}")
    print(f"  Std IoU:           {stab['std_iou']:.4f}")
    print(f"  Comparisons:       {stab['n_comparisons']}")


def plot_sparsity_curve(sparsity_results: dict, labels: list, save_path: str = None):
    """Plot sparsity vs accuracy curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9467bd', '#ff7f0e', '#8c564b']

    for idx, (name, sparsity) in enumerate(sparsity_results.items()):
        color = colors[idx % len(colors)]
        ax.plot(sparsity['percentiles'], sparsity['accuracies'],
                'o-', color=color, label=name, linewidth=2, markersize=8)

    ax.set_xlabel('% of Top Edges Kept', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Sparsity vs Accuracy: How Many Edges Are Needed?', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved sparsity curve to {save_path}")
    plt.close(fig)

    return fig


def plot_validation_summary(all_metrics: dict, save_path: str = None):
    """Plot summary of validation metrics."""
    names = list(all_metrics.keys())
    n = len(names)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9467bd', '#ff7f0e', '#8c564b'][:n]

    # Fidelity+ comparison
    ax1 = axes[0]
    fid_plus = [all_metrics[c]['fidelity_plus']['fidelity_plus'] for c in names]
    bars1 = ax1.bar(range(n), fid_plus, color=colors)
    ax1.set_ylabel('Fidelity+ Score')
    ax1.set_title('Fidelity+ (Sufficiency)\nHigher = removing important edges hurts more')
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    for bar, val in zip(bars1, fid_plus):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Fidelity- comparison
    ax2 = axes[1]
    fid_minus = [all_metrics[c]['fidelity_minus']['fidelity_minus'] for c in names]
    bars2 = ax2.bar(range(n), fid_minus, color=colors)
    ax2.set_ylabel('Fidelity- Score')
    ax2.set_title('Fidelity- (Comprehensiveness)\nHigher = sparse subgraph maintains accuracy')
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim(0, 1.1)
    for bar, val in zip(bars2, fid_minus):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Stability comparison
    ax3 = axes[2]
    stability = [all_metrics[c]['stability']['mean_iou'] for c in names]
    stability_std = [all_metrics[c]['stability']['std_iou'] for c in names]
    bars3 = ax3.bar(range(n), stability, yerr=stability_std, capsize=5, color=colors)
    ax3.set_ylabel('Mean IoU')
    ax3.set_title('Stability (Consistency)\nHigher = more consistent explanations')
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax3.set_ylim(0, 1.1)
    for bar, val in zip(bars3, stability):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved validation summary to {save_path}")
    plt.close(fig)

    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # Configuration
    THRESHOLD = 0.0
    N_SAMPLES_PER_CLASS = 5000
    N_LINES = 10
    BATCH_SIZE = 256
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    MODEL_PATH = "ckpts/dgcnn_seed_model.pth"
    OUTPUT_DIR = "plots"

    ROOT_PATH = "/Users/urbansirca/datasets/SEED/Preprocessed_EEG"
    IO_PATH = "/Users/urbansirca/Desktop/FAX/Master's AI/MLGraphs/DGCNN/.torcheeg/datasets_1768912535105_zLBYu"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    print("\nLoading SEED Dataset...")

    dataset = SEEDDataset(
        root_path=ROOT_PATH,
        io_path=IO_PATH,
        offline_transform=transforms.BandDifferentialEntropy(
            band_dict={
                "delta": [1, 4],
                "theta": [4, 8],
                "alpha": [8, 14],
                "beta": [14, 31],
                "gamma": [31, 49],
            }
        ),
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Compose(
            [
                transforms.Select("emotion"),
                transforms.Lambda(lambda x: x + 1),
            ]
        ),
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0)
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load trained model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = DGCNN(
        in_channels=5,
        num_electrodes=62,
        hid_channels=32,
        num_layers=2,
        num_classes=3,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    # Visualize learned adjacency
    print("\nVisualizing learned adjacency matrix...")
    visualize_learned_adjacency(model, threshold=THRESHOLD, save_path=f"{OUTPUT_DIR}/learned_adjacency.png")

    # Prepare PyG-compatible model
    print("\nPreparing PyG-compatible model for GNNExplainer...")
    explainer_model, edge_index, edge_attr = prepare_for_explainer_pyg(
        model, threshold=THRESHOLD
    )

    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge attr shape: {edge_attr.shape}")
    print(f"Number of edges (including self-loops): {edge_index.shape[1]}")

    # Count self-loops
    num_self_loops = (edge_index[0] == edge_index[1]).sum().item()
    print(f"Number of self-loops: {num_self_loops}")
    print(f"Number of regular edges: {edge_index.shape[1] - num_self_loops}")

    # Get normalized edge weights
    edge_weight = explainer_model._get_normalized_edge_weights(edge_index)

    # Setup GNNExplainer
    print("\nSetting up GNNExplainer...")
    explainer = Explainer(
        model=explainer_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    # =========================================================================
    # STANDARD GNN EXPLAINER
    # =========================================================================

    print("\n" + "="*70)
    print("STANDARD GNN EXPLAINER")
    print("="*70)

    print("\nGenerating aggregated explanations...")
    aggregated = get_aggregated_explanations(
        explainer=explainer,
        data_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_samples_per_class=N_SAMPLES_PER_CLASS,
        num_classes=3,
    )

    class_names = ["Negative", "Neutral", "Positive"]

    # Visualize aggregated results - ALL CLASSES IN SUBPLOTS
    # Edge importance subplots (circular)
    edge_masks_dict = {idx: data['edge_mask_mean'] for idx, data in aggregated.items()}
    visualize_edge_importance_circular_subplots(
        edge_index=edge_index,
        edge_masks=edge_masks_dict,
        class_names=class_names,
        num_electrodes=62,
        n_lines=N_LINES,
        title="Average Edge Importance by Class (Standard)",
        save_path=f"{OUTPUT_DIR}/avg_edge_importance_all_classes.png",
    )

    # Node importance subplots
    node_masks_dict = {idx: data['node_mask_mean'] for idx, data in aggregated.items()}
    visualize_node_importance_subplots(
        node_masks=node_masks_dict,
        class_names=class_names,
        title="Average Node Importance by Class (Standard)",
        save_path=f"{OUTPUT_DIR}/avg_node_importance_all_classes.png",
    )

    # MNE Topographic plots for standard explainer
    print("\nGenerating MNE topographic plots (Standard)...")
    plot_topomap_node_importance_subplots(
        node_masks=node_masks_dict,
        class_names=class_names,
        title="Node Importance - MNE Topomap (Standard)",
        save_path=f"{OUTPUT_DIR}/topomap_node_standard.png",
        normalize_global=True,
    )

    plot_topomap_edge_importance_subplots(
        edge_masks=edge_masks_dict,
        class_names=class_names,
        edge_index=edge_index,
        title="Edge Importance - MNE Topomap (Standard)",
        save_path=f"{OUTPUT_DIR}/topomap_edge_standard.png",
        top_k=N_LINES,
        normalize_global=True,
    )

    # Frequency band topomaps
    for idx, data in aggregated.items():
        node_mask = data['node_mask_mean']
        if node_mask.dim() == 2:
            plot_frequency_band_topomaps(
                node_mask,
                title=f"Frequency Band Importance - {class_names[idx]} (Standard)",
                save_path=f"{OUTPUT_DIR}/topomap_freq_bands_{class_names[idx].lower()}.png",
            )
            break  # Just plot one class as example

    # Validation metrics for standard explainer
    print("\n" + "="*70)
    print("VALIDATION METRICS - STANDARD GNN EXPLAINER")
    print("="*70)

    # Collect samples and masks for validation
    standard_samples_by_class = {i: [] for i in range(3)}
    standard_masks_by_class = {i: [] for i in range(3)}

    for x, y in test_loader:
        for i in range(len(y)):
            label = y[i].item()
            if len(standard_samples_by_class[label]) < N_SAMPLES_PER_CLASS:
                sample = x[i].squeeze()
                explanation = explainer(
                    x=sample,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
                standard_samples_by_class[label].append((sample, label))
                standard_masks_by_class[label].append(explanation.edge_mask.detach().cpu())

        if all(len(v) >= N_SAMPLES_PER_CLASS for v in standard_samples_by_class.values()):
            break

    all_standard_metrics = {}
    for class_idx, class_name in enumerate(class_names):
        if len(standard_samples_by_class[class_idx]) > 0:
            metrics = compute_all_validation_metrics(
                explainer_model,
                standard_samples_by_class[class_idx],
                standard_masks_by_class[class_idx],
                edge_index,
                device=str(DEVICE),
            )
            all_standard_metrics[class_name] = metrics
            print_validation_metrics(metrics, f"Standard - {class_name}")

    # Plot validation summary for standard
    if all_standard_metrics:
        plot_validation_summary(
            all_standard_metrics,
            save_path=f"{OUTPUT_DIR}/validation_summary_standard.png"
        )

        sparsity_results = {name: m['sparsity_curve'] for name, m in all_standard_metrics.items()}
        plot_sparsity_curve(
            sparsity_results,
            list(all_standard_metrics.keys()),
            save_path=f"{OUTPUT_DIR}/sparsity_curve_standard.png"
        )

    # =========================================================================
    # CONTRASTIVE GNN EXPLAINER
    # =========================================================================

    print("\n" + "="*70)
    print("CONTRASTIVE GNN EXPLAINER")
    print("="*70)

    # Define class pairs to contrast
    contrasts = [
        (0, 2, "Negative vs Positive"),
        (2, 0, "Positive vs Negative"),
        (0, 1, "Negative vs Neutral"),
        (1, 0, "Neutral vs Negative"),
        (1, 2, "Neutral vs Positive"),
        (2, 1, "Positive vs Neutral"),
    ]

    # Collect multiple samples per class for contrastive validation
    contrastive_samples_per_class = 10
    class_samples_list = {0: [], 1: [], 2: []}
    for x, y in test_loader:
        for i in range(len(y)):
            label = y[i].item()
            if len(class_samples_list[label]) < contrastive_samples_per_class:
                class_samples_list[label].append(x[i])
        if all(len(v) >= contrastive_samples_per_class for v in class_samples_list.values()):
            break

    # Generate contrastive explanations and collect masks
    contrastive_node_masks = {}
    contrastive_edge_masks = {}
    contrast_labels = []
    contrastive_validation_data = {}

    for target_class, contrast_class, description in contrasts:
        print(f"\n{'='*60}")
        print(f"Contrastive Explanation: {description}")
        print(f"  Target class: {class_names[target_class]}")
        print(f"  Contrast class: {class_names[contrast_class]}")
        print(f"{'='*60}")

        key = f"{target_class}_vs_{contrast_class}"
        contrast_labels.append(description)

        # Generate explanations for multiple samples
        samples_for_validation = []
        edge_masks_for_validation = []

        for sample_idx, sample in enumerate(class_samples_list[target_class]):
            sample = sample.squeeze()

            explanation = explain_class_contrast(
                explainer_model=explainer_model,
                sample=sample,
                edge_index=edge_index,
                edge_weight=edge_weight,
                target_class=target_class,
                contrast_class=contrast_class,
                epochs=200,
                contrast_weight=1.0,
            )

            samples_for_validation.append((sample, target_class))
            edge_masks_for_validation.append(explanation.edge_mask.detach().cpu())

            # Store first sample's masks for visualization
            if sample_idx == 0:
                contrastive_node_masks[key] = explanation.node_mask.squeeze()
                contrastive_edge_masks[key] = explanation.edge_mask

                print(f"  Node mask shape: {explanation.node_mask.shape}")
                print(f"  Edge mask shape: {explanation.edge_mask.shape}")

        contrastive_validation_data[description] = {
            'samples': samples_for_validation,
            'edge_masks': edge_masks_for_validation,
        }

    # Visualize all contrastive explanations as subplots
    print("\nGenerating contrastive subplot visualizations...")

    # Edge importance subplots (circular)
    visualize_edge_importance_circular_subplots(
        edge_index=edge_index,
        edge_masks=contrastive_edge_masks,
        class_names=contrast_labels,
        num_electrodes=62,
        n_lines=N_LINES,
        title="Contrastive Edge Importance",
        save_path=f"{OUTPUT_DIR}/contrastive_edge_all.png",
        normalize_per_plot=True,
    )

    # Node importance subplots
    visualize_node_importance_subplots(
        node_masks=contrastive_node_masks,
        class_names=contrast_labels,
        title="Contrastive Node Importance",
        save_path=f"{OUTPUT_DIR}/contrastive_node_all.png",
        normalize_per_plot=True,
    )

    # MNE Topographic plots for contrastive
    print("\nGenerating MNE topographic plots (Contrastive)...")
    plot_topomap_node_importance_subplots(
        node_masks=contrastive_node_masks,
        class_names=contrast_labels,
        title="Node Importance - MNE Topomap (Contrastive)",
        save_path=f"{OUTPUT_DIR}/topomap_node_contrastive.png",
        normalize_global=False,  # Normalize per plot for contrastive
    )

    plot_topomap_edge_importance_subplots(
        edge_masks=contrastive_edge_masks,
        class_names=contrast_labels,
        edge_index=edge_index,
        title="Edge Importance - MNE Topomap (Contrastive)",
        save_path=f"{OUTPUT_DIR}/topomap_edge_contrastive.png",
        top_k=N_LINES,
        normalize_global=False,
    )

    # Validation metrics for contrastive explainer
    print("\n" + "="*70)
    print("VALIDATION METRICS - CONTRASTIVE GNN EXPLAINER")
    print("="*70)

    all_contrastive_metrics = {}
    for description, data in contrastive_validation_data.items():
        metrics = compute_all_validation_metrics(
            explainer_model,
            data['samples'],
            data['edge_masks'],
            edge_index,
            device=str(DEVICE),
        )
        all_contrastive_metrics[description] = metrics
        print_validation_metrics(metrics, f"Contrastive - {description}")

    # Plot validation summary for contrastive
    if all_contrastive_metrics:
        plot_validation_summary(
            all_contrastive_metrics,
            save_path=f"{OUTPUT_DIR}/validation_summary_contrastive.png"
        )

        sparsity_results = {name: m['sparsity_curve'] for name, m in all_contrastive_metrics.items()}
        plot_sparsity_curve(
            sparsity_results,
            list(all_contrastive_metrics.keys()),
            save_path=f"{OUTPUT_DIR}/sparsity_curve_contrastive.png"
        )

    # =========================================================================
    # COMPARISON: STANDARD vs CONTRASTIVE
    # =========================================================================

    print("\n" + "="*70)
    print("COMPARISON: STANDARD vs CONTRASTIVE")
    print("="*70)

    # Combine metrics for comparison plot
    combined_metrics = {}
    for name, metrics in all_standard_metrics.items():
        combined_metrics[f"Std-{name}"] = metrics
    for name, metrics in all_contrastive_metrics.items():
        # Shorten contrastive names for readability
        short_name = name.replace(" vs ", "/").replace("Negative", "Neg").replace("Positive", "Pos").replace("Neutral", "Neu")
        combined_metrics[f"Con-{short_name}"] = metrics

    if combined_metrics:
        plot_validation_summary(
            combined_metrics,
            save_path=f"{OUTPUT_DIR}/validation_summary_comparison.png"
        )

    # Save all metrics to JSON
    metrics_summary = {
        'standard': {},
        'contrastive': {}
    }

    for name, metrics in all_standard_metrics.items():
        metrics_summary['standard'][name] = {
            'fidelity_plus': metrics['fidelity_plus']['fidelity_plus'],
            'fidelity_minus': metrics['fidelity_minus']['fidelity_minus'],
            'stability_mean_iou': metrics['stability']['mean_iou'],
            'stability_std_iou': metrics['stability']['std_iou'],
        }

    for name, metrics in all_contrastive_metrics.items():
        metrics_summary['contrastive'][name] = {
            'fidelity_plus': metrics['fidelity_plus']['fidelity_plus'],
            'fidelity_minus': metrics['fidelity_minus']['fidelity_minus'],
            'stability_mean_iou': metrics['stability']['mean_iou'],
            'stability_std_iou': metrics['stability']['std_iou'],
        }

    with open(f"{OUTPUT_DIR}/validation_metrics.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved validation metrics to {OUTPUT_DIR}/validation_metrics.json")

    print("\n" + "="*70)
    print("EXPLANATION COMPLETE")
    print("="*70)
    print(f"All results saved to: {OUTPUT_DIR}/")

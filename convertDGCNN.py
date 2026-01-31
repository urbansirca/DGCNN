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


# ============================================================================
# CLASS-LEVEL PROTOTYPE EXPLANATIONS (GNNExplainer Paper Method)
# ============================================================================

def get_embedding(
    model: DGCNNForExplainerPyG,
    sample: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Extract embedding before final classification layer.

    This returns the flattened output after graph convolution but before
    the fully connected classification layers.

    Args:
        model: The PyG-compatible DGCNN model
        sample: Input features [num_nodes, num_features] or [1, num_nodes, num_features]
        edge_index: Edge indices [2, num_edges]
        edge_weight: Edge weights [num_edges]

    Returns:
        Embedding tensor of shape [num_electrodes * hid_channels]
    """
    # Handle input shape
    if sample.dim() == 3:
        sample = sample.squeeze(0)

    # Apply batch norm
    x = sample.unsqueeze(0)
    x = model.BN1(x.transpose(1, 2)).transpose(1, 2)
    x = x.squeeze(0)

    # Get self-loop mask
    self_loop_mask = model._get_self_loop_mask(edge_index)

    # Graph convolution
    result = model.layer1(x, edge_index, edge_weight, model.num_electrodes, self_loop_mask)

    # Flatten (this is the "embedding" before fc layers)
    embedding = result.reshape(-1)

    return embedding


def get_class_prototype_explanation(
    model: DGCNNForExplainerPyG,
    explainer,
    data_loader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    target_class: int,
    num_samples: int = 100,
    use_contrastive: bool = False,
    contrast_class: int = None,
    contrast_weight: float = 1.0,
):
    """
    Generate class-level prototype explanation following the GNNExplainer paper.

    This implements the 4-step process described in the original paper:
    1. Find reference instance (embedding closest to class mean)
    2. Compute explanations for many instances in the class
    3. Align explanation graphs to reference (trivial for fixed EEG topology)
    4. Aggregate with median for robustness to outliers

    Args:
        model: The PyG-compatible DGCNN model
        explainer: GNNExplainer instance (used if use_contrastive=False)
        data_loader: DataLoader for the dataset
        edge_index: Edge indices [2, num_edges]
        edge_weight: Edge weights [num_edges]
        target_class: The class to explain
        num_samples: Number of samples to use for prototype computation
        use_contrastive: If True, use contrastive explanations
        contrast_class: Class to contrast against (required if use_contrastive=True)
        contrast_weight: Weight for contrastive term

    Returns:
        Dictionary containing:
        - prototype_node_mask: Median-aggregated node importance [62, 5]
        - prototype_edge_mask: Median-aggregated edge importance [num_edges]
        - reference_sample: The reference instance
        - reference_idx: Index of reference instance
        - reference_embedding: Embedding of reference instance
        - mean_embedding: Mean embedding of the class
        - num_samples: Number of samples used
        - mean_node_mask: Mean-aggregated node importance (for comparison)
        - mean_edge_mask: Mean-aggregated edge importance (for comparison)
        - all_node_masks: All individual node masks [N, 62, 5]
        - all_edge_masks: All individual edge masks [N, num_edges]
    """
    model.eval()

    print(f"\n{'='*60}")
    print(f"Computing Class Prototype Explanation for Class {target_class}")
    print(f"{'='*60}")

    # Step 1: Collect samples and embeddings for the target class
    print(f"\nStep 1: Collecting {num_samples} samples and computing embeddings...")

    embeddings = []
    samples = []

    for x, y in data_loader:
        for i in range(len(y)):
            if y[i].item() == target_class:
                sample = x[i].squeeze()
                samples.append(sample)

                # Get embedding (before final classification layer)
                with torch.no_grad():
                    emb = get_embedding(model, sample, edge_index, edge_weight)
                embeddings.append(emb)

                print(f"  Collected {len(embeddings)}/{num_samples} samples", end="\r")

                if len(embeddings) >= num_samples:
                    break
        if len(embeddings) >= num_samples:
            break

    print(f"\n  Collected {len(embeddings)} samples")

    if len(embeddings) == 0:
        raise ValueError(f"No samples found for class {target_class}")

    # Compute mean embedding
    embeddings = torch.stack(embeddings)  # [N, emb_dim]
    mean_embedding = embeddings.mean(dim=0)

    # Find reference instance (closest to mean)
    distances = torch.norm(embeddings - mean_embedding, dim=1)
    reference_idx = distances.argmin().item()
    reference_sample = samples[reference_idx]
    reference_embedding = embeddings[reference_idx]

    print(f"\nStep 2: Reference instance selected")
    print(f"  Reference index: {reference_idx}")
    print(f"  Distance to mean: {distances[reference_idx]:.4f}")
    print(f"  Mean distance: {distances.mean():.4f}")
    print(f"  Max distance: {distances.max():.4f}")

    # Step 2 & 3: Compute explanations for all samples
    # (Alignment is trivial for fixed EEG graph structure - nodes already correspond)
    print(f"\nStep 3: Computing explanations for {len(samples)} samples...")

    node_masks = []
    edge_masks = []

    for idx, sample in enumerate(samples):
        if use_contrastive and contrast_class is not None:
            explanation = explain_class_contrast(
                explainer_model=model,
                sample=sample,
                edge_index=edge_index,
                edge_weight=edge_weight,
                target_class=target_class,
                contrast_class=contrast_class,
                epochs=200,
                contrast_weight=contrast_weight,
            )
        else:
            explanation = explainer(
                x=sample,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

        node_masks.append(explanation.node_mask.detach().cpu())
        edge_masks.append(explanation.edge_mask.detach().cpu())

        print(f"  Computed {idx + 1}/{len(samples)} explanations", end="\r")

    print(f"\n  Completed {len(samples)} explanations")

    # Step 4: Aggregate with MEDIAN (robust to outliers)
    print(f"\nStep 4: Aggregating explanations with median...")

    node_masks = torch.stack(node_masks)  # [N, 62, 5]
    edge_masks = torch.stack(edge_masks)  # [N, num_edges]

    # Median aggregation (robust to outlier explanations)
    prototype_node_mask = torch.median(node_masks, dim=0).values
    prototype_edge_mask = torch.median(edge_masks, dim=0).values

    # Also compute mean for comparison
    mean_node_mask = node_masks.mean(dim=0)
    mean_edge_mask = edge_masks.mean(dim=0)

    # Compute standard deviation for uncertainty quantification
    std_node_mask = node_masks.std(dim=0)
    std_edge_mask = edge_masks.std(dim=0)

    print(f"  Prototype node mask shape: {prototype_node_mask.shape}")
    print(f"  Prototype edge mask shape: {prototype_edge_mask.shape}")

    # Compute agreement statistics
    node_agreement = 1 - (std_node_mask / (mean_node_mask.abs() + 1e-10)).mean()
    edge_agreement = 1 - (std_edge_mask / (mean_edge_mask.abs() + 1e-10)).mean()
    print(f"  Node mask agreement (1 - CV): {node_agreement:.4f}")
    print(f"  Edge mask agreement (1 - CV): {edge_agreement:.4f}")

    return {
        'prototype_node_mask': prototype_node_mask,
        'prototype_edge_mask': prototype_edge_mask,
        'reference_sample': reference_sample,
        'reference_idx': reference_idx,
        'reference_embedding': reference_embedding,
        'mean_embedding': mean_embedding,
        'num_samples': len(samples),
        'mean_node_mask': mean_node_mask,
        'mean_edge_mask': mean_edge_mask,
        'std_node_mask': std_node_mask,
        'std_edge_mask': std_edge_mask,
        'all_node_masks': node_masks,
        'all_edge_masks': edge_masks,
        'embedding_distances': distances,
    }


def get_all_class_prototypes(
    model: DGCNNForExplainerPyG,
    explainer,
    data_loader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_classes: int = 3,
    num_samples_per_class: int = 100,
    use_contrastive: bool = False,
    contrast_weight: float = 1.0,
):
    """
    Compute class prototype explanations for all classes.

    Args:
        model: The PyG-compatible DGCNN model
        explainer: GNNExplainer instance
        data_loader: DataLoader for the dataset
        edge_index: Edge indices [2, num_edges]
        edge_weight: Edge weights [num_edges]
        num_classes: Number of classes
        num_samples_per_class: Number of samples per class
        use_contrastive: If True, use contrastive explanations
        contrast_weight: Weight for contrastive term

    Returns:
        Dictionary mapping class_idx to prototype explanation dict
    """
    prototypes = {}

    for class_idx in range(num_classes):
        # For contrastive, use the "most different" class as contrast
        # (e.g., for emotion: Negative contrasts with Positive)
        if use_contrastive:
            if class_idx == 0:  # Negative
                contrast_class = 2  # Positive
            elif class_idx == 2:  # Positive
                contrast_class = 0  # Negative
            else:  # Neutral
                contrast_class = 0  # Contrast with Negative
        else:
            contrast_class = None

        prototype = get_class_prototype_explanation(
            model=model,
            explainer=explainer,
            data_loader=data_loader,
            edge_index=edge_index,
            edge_weight=edge_weight,
            target_class=class_idx,
            num_samples=num_samples_per_class,
            use_contrastive=use_contrastive,
            contrast_class=contrast_class,
            contrast_weight=contrast_weight,
        )
        prototypes[class_idx] = prototype

    return prototypes


def visualize_prototype_comparison(
    prototypes: dict,
    edge_index: torch.Tensor,
    class_names: list,
    num_electrodes: int = 62,
    n_lines: int = 50,
    save_dir: str = "plots",
):
    """
    Visualize and compare prototype explanations: median vs mean aggregation.

    Args:
        prototypes: Dictionary from get_all_class_prototypes
        edge_index: Edge indices
        class_names: List of class names
        num_electrodes: Number of electrodes
        n_lines: Number of top edges to show
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prepare masks for visualization
    median_node_masks = {idx: p['prototype_node_mask'] for idx, p in prototypes.items()}
    median_edge_masks = {idx: p['prototype_edge_mask'] for idx, p in prototypes.items()}
    mean_node_masks = {idx: p['mean_node_mask'] for idx, p in prototypes.items()}
    mean_edge_masks = {idx: p['mean_edge_mask'] for idx, p in prototypes.items()}

    # Visualize median (prototype) node importance
    visualize_node_importance_subplots(
        node_masks=median_node_masks,
        class_names=class_names,
        title="Class Prototype Node Importance (Median Aggregation)",
        save_path=f"{save_dir}/prototype_node_median.png",
    )

    # Visualize mean node importance for comparison
    visualize_node_importance_subplots(
        node_masks=mean_node_masks,
        class_names=class_names,
        title="Class Node Importance (Mean Aggregation)",
        save_path=f"{save_dir}/prototype_node_mean.png",
    )

    # Visualize median edge importance (circular)
    visualize_edge_importance_circular_subplots(
        edge_index=edge_index,
        edge_masks=median_edge_masks,
        class_names=class_names,
        num_electrodes=num_electrodes,
        n_lines=n_lines,
        title="Class Prototype Edge Importance (Median Aggregation)",
        save_path=f"{save_dir}/prototype_edge_median.png",
    )

    # Visualize mean edge importance for comparison (circular)
    visualize_edge_importance_circular_subplots(
        edge_index=edge_index,
        edge_masks=mean_edge_masks,
        class_names=class_names,
        num_electrodes=num_electrodes,
        n_lines=n_lines,
        title="Class Edge Importance (Mean Aggregation)",
        save_path=f"{save_dir}/prototype_edge_mean.png",
    )

    # NEW: Arrow-based edge importance visualization for prototypes
    plot_edge_importance_arrows_subplots(
        edge_index=edge_index,
        edge_masks=median_edge_masks,
        class_names=class_names,
        num_electrodes=num_electrodes,
        top_k=n_lines,
        title="Class Prototype Edge Importance (Median) - Arrow Plot",
        save_path=f"{save_dir}/prototype_edge_arrows_median.png",
        cmap_name='hot',
        node_size=300,
        display_labels=True,
        threshold_percentile=85,
        directed=False,
        normalize_global=True,
    )

    plot_edge_importance_arrows_subplots(
        edge_index=edge_index,
        edge_masks=mean_edge_masks,
        class_names=class_names,
        num_electrodes=num_electrodes,
        top_k=n_lines,
        title="Class Edge Importance (Mean) - Arrow Plot",
        save_path=f"{save_dir}/prototype_edge_arrows_mean.png",
        cmap_name='hot',
        node_size=300,
        display_labels=True,
        threshold_percentile=85,
        directed=False,
        normalize_global=True,
    )

    # MNE topomaps
    plot_topomap_node_importance_subplots(
        node_masks=median_node_masks,
        class_names=class_names,
        title="Class Prototype - MNE Topomap (Median)",
        save_path=f"{save_dir}/prototype_topomap_median.png",
    )

    plot_topomap_node_importance_subplots(
        node_masks=mean_node_masks,
        class_names=class_names,
        title="Class - MNE Topomap (Mean)",
        save_path=f"{save_dir}/prototype_topomap_mean.png",
    )

    print(f"\nPrototype visualizations saved to {save_dir}/")


def visualize_prototype_uncertainty(
    prototypes: dict,
    class_names: list,
    save_dir: str = "plots",
):
    """
    Visualize uncertainty/variance in prototype explanations.

    Shows how consistent the explanations are across instances.
    Lower variance = more reliable prototype.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_classes = len(prototypes)
    fig, axes = plt.subplots(2, num_classes, figsize=(5 * num_classes, 10))

    for idx, (class_idx, proto) in enumerate(prototypes.items()):
        class_name = class_names[idx] if idx < len(class_names) else str(class_idx)

        # Node mask variance
        ax_node = axes[0, idx] if num_classes > 1 else axes[0]
        node_std = proto['std_node_mask'].mean(dim=-1).numpy()  # Average across features

        positions = get_electrode_positions(62)
        scatter = ax_node.scatter(
            positions[:, 0], positions[:, 1],
            c=node_std, cmap='Blues', s=200,
            edgecolors='black', linewidths=1,
        )
        ax_node.set_title(f"{class_name}\nNode Importance Std Dev")
        ax_node.set_aspect('equal')
        ax_node.axis('off')
        plt.colorbar(scatter, ax=ax_node, shrink=0.7)

        # Edge mask histogram
        ax_edge = axes[1, idx] if num_classes > 1 else axes[1]
        edge_std = proto['std_edge_mask'].numpy()
        ax_edge.hist(edge_std, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax_edge.set_xlabel('Edge Importance Std Dev')
        ax_edge.set_ylabel('Count')
        ax_edge.set_title(f"{class_name}\nEdge Importance Variance Distribution")

    plt.suptitle("Prototype Explanation Uncertainty", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/prototype_uncertainty.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Uncertainty visualization saved to {save_dir}/prototype_uncertainty.png")


def visualize_embedding_space(
    prototypes: dict,
    class_names: list,
    save_dir: str = "plots",
):
    """
    Visualize the embedding space and reference instance selection.

    Uses t-SNE to project embeddings to 2D and shows:
    - All instance embeddings colored by class
    - Reference instances highlighted
    - Class centroids
    """
    from sklearn.manifold import TSNE

    os.makedirs(save_dir, exist_ok=True)

    # Collect all embeddings and reference info
    all_embeddings = []
    all_labels = []
    reference_indices = []

    current_idx = 0
    for class_idx, proto in prototypes.items():
        n_samples = proto['num_samples']

        # Reconstruct embeddings from distances (approximate)
        # Actually, we need to store embeddings - let's use the stored ones
        # For now, we'll create a simplified visualization
        reference_indices.append(current_idx + proto['reference_idx'])
        current_idx += n_samples

    # Since we don't have all embeddings stored, let's visualize distances instead
    fig, axes = plt.subplots(1, len(prototypes), figsize=(5 * len(prototypes), 5))
    if len(prototypes) == 1:
        axes = [axes]

    for idx, (class_idx, proto) in enumerate(prototypes.items()):
        ax = axes[idx]
        class_name = class_names[idx] if idx < len(class_names) else str(class_idx)

        distances = proto['embedding_distances'].numpy()
        ref_idx = proto['reference_idx']

        # Plot histogram of distances
        ax.hist(distances, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(distances[ref_idx], color='red', linestyle='--', linewidth=2,
                   label=f'Reference (idx={ref_idx})')
        ax.axvline(distances.mean(), color='green', linestyle=':', linewidth=2,
                   label=f'Mean dist={distances.mean():.3f}')
        ax.set_xlabel('Distance to Class Centroid')
        ax.set_ylabel('Count')
        ax.set_title(f"{class_name}\nEmbedding Distances")
        ax.legend()

    plt.suptitle("Reference Instance Selection: Distance to Class Centroid", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/prototype_embedding_distances.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Embedding distance visualization saved to {save_dir}/prototype_embedding_distances.png")


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


def plot_edge_importance_arrows(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    num_electrodes: int = 62,
    electrode_names: list = None,
    top_k: int = 50,
    title: str = "Edge Importance",
    save_path: str = None,
    ax: plt.Axes = None,
    cmap_name: str = 'hot',
    node_size: int = 400,
    display_labels: bool = True,
    threshold_percentile: float = 90,
    directed: bool = False,
):
    """
    Visualize edge importance using curved arrows on anatomically correct head layout.

    Uses FancyArrowPatch for cleaner, more interpretable edge visualization.
    Node colors represent aggregated incoming edge importance (how much attention flows TO each node).

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_mask: Edge importance values [num_edges]
        num_electrodes: Number of electrodes
        electrode_names: List of electrode names
        top_k: Number of top edges to display
        title: Plot title
        save_path: Path to save figure
        ax: Matplotlib axis (creates new figure if None)
        cmap_name: Colormap name
        node_size: Size of electrode nodes
        display_labels: Whether to show electrode labels
        threshold_percentile: Only show edges above this percentile
        directed: If True, show directed arrows; if False, show undirected lines

    Returns:
        Matplotlib axis
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES[:num_electrodes]

    # Get MNE positions
    info = get_mne_info()
    pos = np.array([info['chs'][i]['loc'][:2] for i in range(len(info['chs']))])

    # Convert edge_mask to numpy
    if isinstance(edge_mask, torch.Tensor):
        edge_mask = edge_mask.cpu().numpy()
    edge_idx_np = edge_index.cpu().numpy()

    # Build adjacency matrix from edge_index and edge_mask
    adj_matrix = np.zeros((num_electrodes, num_electrodes))
    for i in range(edge_idx_np.shape[1]):
        src, dst = edge_idx_np[0, i], edge_idx_np[1, i]
        if src != dst:  # Skip self-loops
            adj_matrix[src, dst] = edge_mask[i]

    # For undirected visualization, symmetrize
    if not directed:
        adj_matrix = (adj_matrix + adj_matrix.T) / 2

    # Compute node importance as sum of incoming edge weights
    node_importance = adj_matrix.sum(axis=0)  # Sum of incoming edges
    node_importance = node_importance / (node_importance.max() + 1e-10)  # Normalize

    # Thresholding
    threshold = np.percentile(adj_matrix[adj_matrix > 0], threshold_percentile) if threshold_percentile > 0 else 0

    # Color setup
    cmap = plt.get_cmap(cmap_name)
    v_min = threshold
    v_max = adj_matrix.max()

    # Create figure if needed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        created_fig = True

    # Draw head outline
    head_radius = 0.1
    head_center = (0, 0)
    circle = plt.Circle(head_center, head_radius, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    # Nose
    ax.plot([0, 0.015, 0], [head_radius, head_radius * 1.08, head_radius], 'k-', linewidth=2)
    # Ears
    ax.plot([-head_radius, -head_radius * 1.05, -head_radius], [-0.008, 0, 0.008], 'k-', linewidth=2)
    ax.plot([head_radius, head_radius * 1.05, head_radius], [-0.008, 0, 0.008], 'k-', linewidth=2)

    # Get top-k edges
    flat_adj = adj_matrix.flatten()
    top_indices = np.argsort(flat_adj)[-top_k:]
    top_edges = np.unravel_index(top_indices, adj_matrix.shape)

    # Draw edges as curved arrows
    drawn_edges = set()
    for src, dst in zip(top_edges[0], top_edges[1]):
        if src == dst:
            continue

        # For undirected, only draw each edge once
        if not directed:
            edge_key = (min(src, dst), max(src, dst))
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)

        w = adj_matrix[src, dst]
        if w < threshold:
            continue

        # Normalize weight for color and linewidth
        w_norm = (w - v_min) / (v_max - v_min + 1e-10)
        color = cmap(w_norm)
        linewidth = 0.5 + 4 * w_norm

        if directed:
            # Draw directed arrow
            arrow = FancyArrowPatch(
                posA=pos[src],
                posB=pos[dst],
                connectionstyle="arc3,rad=0.15",
                arrowstyle='-|>',
                mutation_scale=12,
                color=color,
                linewidth=linewidth,
                alpha=0.4 + 0.6 * w_norm,
                zorder=2,
                shrinkA=np.sqrt(node_size) / 2.5,
                shrinkB=np.sqrt(node_size) / 2.5,
            )
            ax.add_patch(arrow)
        else:
            # Draw undirected curved line
            arrow = FancyArrowPatch(
                posA=pos[src],
                posB=pos[dst],
                connectionstyle="arc3,rad=0.1",
                arrowstyle='-',
                color=color,
                linewidth=linewidth,
                alpha=0.4 + 0.6 * w_norm,
                zorder=2,
                shrinkA=np.sqrt(node_size) / 3,
                shrinkB=np.sqrt(node_size) / 3,
            )
            ax.add_patch(arrow)

    # Draw nodes colored by incoming edge importance
    node_colors = [cmap(val) for val in node_importance]
    ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c=node_colors,
               edgecolors='black', linewidths=1.5, zorder=10)

    # Add electrode labels
    if display_labels:
        for i, name in enumerate(electrode_names):
            ax.text(pos[i, 0], pos[i, 1], name, ha='center', va='center',
                    fontsize=6, fontweight='bold', zorder=11)

    ax.set_xlim(-0.14, 0.14)
    ax.set_ylim(-0.14, 0.14)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12)

    # Add colorbar
    if created_fig:
        norm = Normalize(vmin=v_min, vmax=v_max)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Edge Importance', fontsize=10)

    if save_path and created_fig:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved edge importance plot to {save_path}")

    if created_fig:
        plt.close()

    return ax


def plot_edge_importance_arrows_subplots(
    edge_index: torch.Tensor,
    edge_masks: dict,
    class_names: list,
    num_electrodes: int = 62,
    electrode_names: list = None,
    top_k: int = 50,
    title: str = "Edge Importance by Class",
    save_path: str = None,
    cmap_name: str = 'hot',
    node_size: int = 300,
    display_labels: bool = True,
    threshold_percentile: float = 85,
    directed: bool = False,
    normalize_global: bool = True,
    ncols: int = None,
):
    """
    Visualize edge importance for multiple classes using curved arrows.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_masks: Dictionary mapping class_idx/key -> edge_mask tensor
        class_names: List of class names
        num_electrodes: Number of electrodes
        electrode_names: List of electrode names
        top_k: Number of top edges to display per class
        title: Overall figure title
        save_path: Path to save figure
        cmap_name: Colormap name
        node_size: Size of electrode nodes
        display_labels: Whether to show electrode labels
        threshold_percentile: Only show edges above this percentile
        directed: If True, show directed arrows
        normalize_global: If True, use same color scale across all plots
        ncols: Number of columns

    Returns:
        Matplotlib figure
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES[:num_electrodes]

    num_plots = len(edge_masks)
    if ncols is None:
        ncols = min(num_plots, 3)
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))

    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Get MNE positions
    info = get_mne_info()
    pos = np.array([info['chs'][i]['loc'][:2] for i in range(len(info['chs']))])

    edge_idx_np = edge_index.cpu().numpy()
    cmap = plt.get_cmap(cmap_name)

    # Build all adjacency matrices first
    adj_matrices = {}
    global_max = 0
    for key, edge_mask in edge_masks.items():
        if isinstance(edge_mask, torch.Tensor):
            edge_mask = edge_mask.cpu().numpy()

        adj_matrix = np.zeros((num_electrodes, num_electrodes))
        for i in range(edge_idx_np.shape[1]):
            src, dst = edge_idx_np[0, i], edge_idx_np[1, i]
            if src != dst:
                adj_matrix[src, dst] = edge_mask[i]

        if not directed:
            adj_matrix = (adj_matrix + adj_matrix.T) / 2

        adj_matrices[key] = adj_matrix
        global_max = max(global_max, adj_matrix.max())

    # Plot each class
    for ax_idx, (key, adj_matrix) in enumerate(adj_matrices.items()):
        ax = axes[ax_idx]
        label = class_names[ax_idx] if ax_idx < len(class_names) else str(key)

        # Compute node importance
        node_importance = adj_matrix.sum(axis=0)
        node_importance = node_importance / (node_importance.max() + 1e-10)

        # Thresholding
        nonzero_vals = adj_matrix[adj_matrix > 0]
        if len(nonzero_vals) > 0:
            threshold = np.percentile(nonzero_vals, threshold_percentile)
        else:
            threshold = 0

        v_max = global_max if normalize_global else adj_matrix.max()
        v_min = 0

        # Draw head outline
        head_radius = 0.1
        circle = plt.Circle((0, 0), head_radius, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        ax.plot([0, 0.015, 0], [head_radius, head_radius * 1.08, head_radius], 'k-', linewidth=2)
        ax.plot([-head_radius, -head_radius * 1.05, -head_radius], [-0.008, 0, 0.008], 'k-', linewidth=2)
        ax.plot([head_radius, head_radius * 1.05, head_radius], [-0.008, 0, 0.008], 'k-', linewidth=2)

        # Get top-k edges
        flat_adj = adj_matrix.flatten()
        top_indices = np.argsort(flat_adj)[-top_k:]
        top_edges = np.unravel_index(top_indices, adj_matrix.shape)

        # Draw edges
        drawn_edges = set()
        for src, dst in zip(top_edges[0], top_edges[1]):
            if src == dst:
                continue

            if not directed:
                edge_key = (min(src, dst), max(src, dst))
                if edge_key in drawn_edges:
                    continue
                drawn_edges.add(edge_key)

            w = adj_matrix[src, dst]
            if w < threshold:
                continue

            w_norm = (w - v_min) / (v_max - v_min + 1e-10)
            color = cmap(w_norm)
            linewidth = 0.5 + 3 * w_norm

            if directed:
                arrow = FancyArrowPatch(
                    posA=pos[src], posB=pos[dst],
                    connectionstyle="arc3,rad=0.15",
                    arrowstyle='-|>',
                    mutation_scale=10,
                    color=color,
                    linewidth=linewidth,
                    alpha=0.4 + 0.6 * w_norm,
                    zorder=2,
                    shrinkA=np.sqrt(node_size) / 2.5,
                    shrinkB=np.sqrt(node_size) / 2.5,
                )
            else:
                arrow = FancyArrowPatch(
                    posA=pos[src], posB=pos[dst],
                    connectionstyle="arc3,rad=0.1",
                    arrowstyle='-',
                    color=color,
                    linewidth=linewidth,
                    alpha=0.4 + 0.6 * w_norm,
                    zorder=2,
                    shrinkA=np.sqrt(node_size) / 3,
                    shrinkB=np.sqrt(node_size) / 3,
                )
            ax.add_patch(arrow)

        # Draw nodes
        node_colors = [cmap(val) for val in node_importance]
        ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c=node_colors,
                   edgecolors='black', linewidths=1, zorder=10)

        if display_labels:
            for i, name in enumerate(electrode_names):
                ax.text(pos[i, 0], pos[i, 1], name, ha='center', va='center',
                        fontsize=5, fontweight='bold', zorder=11)

        ax.set_xlim(-0.14, 0.14)
        ax.set_ylim(-0.14, 0.14)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(label, fontsize=11)

    # Hide unused axes
    for ax_idx in range(num_plots, len(axes)):
        axes[ax_idx].set_visible(False)

    # Add shared colorbar
    norm = Normalize(vmin=0, vmax=global_max)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Edge Importance', fontsize=10)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved edge importance arrows plot to {save_path}")
    plt.close(fig)

    return fig


# ============================================================================
# REGION-BASED CONNECTIVITY VISUALIZATION
# ============================================================================

# Define brain region groupings for 62-channel SEED layout
BRAIN_REGIONS = {
    'Prefrontal': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4'],
    'Frontal-L': ['F7', 'F5', 'F3', 'F1'],
    'Frontal-M': ['FZ'],
    'Frontal-R': ['F2', 'F4', 'F6', 'F8'],
    'Frontal-Central-L': ['FT7', 'FC5', 'FC3', 'FC1'],
    'Frontal-Central-M': ['FCZ'],
    'Frontal-Central-R': ['FC2', 'FC4', 'FC6', 'FT8'],
    'Temporal-L': ['T7', 'TP7'],
    'Central-L': ['C5', 'C3', 'C1'],
    'Central-M': ['CZ'],
    'Central-R': ['C2', 'C4', 'C6'],
    'Temporal-R': ['T8', 'TP8'],
    'Central-Parietal-L': ['CP5', 'CP3', 'CP1'],
    'Central-Parietal-M': ['CPZ'],
    'Central-Parietal-R': ['CP2', 'CP4', 'CP6'],
    'Parietal-L': ['P7', 'P5', 'P3', 'P1'],
    'Parietal-M': ['PZ'],
    'Parietal-R': ['P2', 'P4', 'P6', 'P8'],
    'Parietal-Occipital-L': ['PO7', 'PO5', 'PO3'],
    'Parietal-Occipital-M': ['POZ'],
    'Parietal-Occipital-R': ['PO4', 'PO6', 'PO8'],
    'Occipital': ['CB1', 'O1', 'OZ', 'O2', 'CB2'],
}

# Simplified region grouping (fewer regions for cleaner visualization)
BRAIN_REGIONS_SIMPLE = {
    'Prefrontal': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4'],
    'Frontal-L': ['F7', 'F5', 'F3', 'F1', 'FT7', 'FC5', 'FC3', 'FC1'],
    'Frontal-M': ['FZ', 'FCZ'],
    'Frontal-R': ['F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 'FC6', 'FT8'],
    'Temporal-L': ['T7', 'TP7'],
    'Central-L': ['C5', 'C3', 'C1', 'CP5', 'CP3', 'CP1'],
    'Central-M': ['CZ', 'CPZ'],
    'Central-R': ['C2', 'C4', 'C6', 'CP2', 'CP4', 'CP6'],
    'Temporal-R': ['T8', 'TP8'],
    'Parietal-L': ['P7', 'P5', 'P3', 'P1', 'PO7', 'PO5', 'PO3'],
    'Parietal-M': ['PZ', 'POZ'],
    'Parietal-R': ['P2', 'P4', 'P6', 'P8', 'PO4', 'PO6', 'PO8'],
    'Occipital': ['CB1', 'O1', 'OZ', 'O2', 'CB2'],
}

# Even simpler: 6 main regions
BRAIN_REGIONS_COARSE = {
    'Frontal': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
                'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
                'FC4', 'FC6', 'FT8'],
    'Temporal-L': ['T7', 'TP7'],
    'Central': ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
                'CPZ', 'CP2', 'CP4', 'CP6'],
    'Temporal-R': ['T8', 'TP8'],
    'Parietal': ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
                 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8'],
    'Occipital': ['CB1', 'O1', 'OZ', 'O2', 'CB2'],
}


def get_electrode_to_region_mapping(
    electrode_names: list = None,
    regions: dict = None,
) -> dict:
    """
    Create a mapping from electrode name to region index.

    Args:
        electrode_names: List of electrode names
        regions: Dictionary mapping region name to list of electrodes

    Returns:
        Dictionary mapping electrode name to (region_name, region_idx)
    """
    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES
    if regions is None:
        regions = BRAIN_REGIONS_SIMPLE

    electrode_to_region = {}
    for region_idx, (region_name, electrodes) in enumerate(regions.items()):
        for elec in electrodes:
            if elec in electrode_names:
                electrode_to_region[elec] = (region_name, region_idx)

    return electrode_to_region


def aggregate_edge_importance_by_region(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    electrode_names: list = None,
    regions: dict = None,
    aggregation: str = 'mean',
) -> tuple:
    """
    Aggregate edge importance from electrode-level to region-level.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_mask: Edge importance values [num_edges]
        electrode_names: List of electrode names
        regions: Dictionary mapping region name to list of electrodes
        aggregation: How to aggregate ('mean', 'sum', 'max')

    Returns:
        Tuple of (region_names, region_connectivity_matrix, region_importance)
    """
    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES
    if regions is None:
        regions = BRAIN_REGIONS_SIMPLE

    # Convert to numpy
    if isinstance(edge_mask, torch.Tensor):
        edge_mask = edge_mask.cpu().numpy()
    edge_idx_np = edge_index.cpu().numpy()

    # Create electrode to region mapping
    electrode_to_region = get_electrode_to_region_mapping(electrode_names, regions)
    region_names = list(regions.keys())
    num_regions = len(region_names)

    # Create electrode index to region index mapping
    elec_idx_to_region_idx = {}
    for i, elec_name in enumerate(electrode_names):
        if elec_name in electrode_to_region:
            _, region_idx = electrode_to_region[elec_name]
            elec_idx_to_region_idx[i] = region_idx

    # Aggregate edges by region pair
    # Store all edge weights for each region pair
    region_edges = {(i, j): [] for i in range(num_regions) for j in range(num_regions)}

    for edge_idx in range(edge_idx_np.shape[1]):
        src, dst = edge_idx_np[0, edge_idx], edge_idx_np[1, edge_idx]
        if src == dst:  # Skip self-loops
            continue
        if src not in elec_idx_to_region_idx or dst not in elec_idx_to_region_idx:
            continue

        src_region = elec_idx_to_region_idx[src]
        dst_region = elec_idx_to_region_idx[dst]
        region_edges[(src_region, dst_region)].append(edge_mask[edge_idx])

    # Aggregate to create region connectivity matrix
    region_matrix = np.zeros((num_regions, num_regions))
    for (src_reg, dst_reg), weights in region_edges.items():
        if len(weights) > 0:
            if aggregation == 'mean':
                region_matrix[src_reg, dst_reg] = np.mean(weights)
            elif aggregation == 'sum':
                region_matrix[src_reg, dst_reg] = np.sum(weights)
            elif aggregation == 'max':
                region_matrix[src_reg, dst_reg] = np.max(weights)

    # Symmetrize for undirected visualization
    region_matrix = (region_matrix + region_matrix.T) / 2

    # Compute region importance (sum of incoming connections)
    region_importance = region_matrix.sum(axis=0)
    region_importance = region_importance / (region_importance.max() + 1e-10)

    return region_names, region_matrix, region_importance


def get_region_positions(regions: dict = None) -> dict:
    """
    Get 2D positions for brain regions on a schematic head layout.

    Returns dictionary mapping region name to (x, y) position.
    """
    if regions is None:
        regions = BRAIN_REGIONS_SIMPLE

    # Define positions for simplified regions (BRAIN_REGIONS_SIMPLE)
    region_positions = {
        'Prefrontal': (0, 0.85),
        'Frontal-L': (-0.4, 0.6),
        'Frontal-M': (0, 0.6),
        'Frontal-R': (0.4, 0.6),
        'Temporal-L': (-0.7, 0.3),
        'Central-L': (-0.35, 0.3),
        'Central-M': (0, 0.3),
        'Central-R': (0.35, 0.3),
        'Temporal-R': (0.7, 0.3),
        'Parietal-L': (-0.4, 0.0),
        'Parietal-M': (0, 0.0),
        'Parietal-R': (0.4, 0.0),
        'Occipital': (0, -0.25),
    }

    # Positions for coarse regions (BRAIN_REGIONS_COARSE)
    coarse_positions = {
        'Frontal': (0, 0.7),
        'Temporal-L': (-0.7, 0.35),
        'Central': (0, 0.35),
        'Temporal-R': (0.7, 0.35),
        'Parietal': (0, 0.0),
        'Occipital': (0, -0.25),
    }

    # Return appropriate positions based on region names
    region_names = list(regions.keys())
    if 'Central' in region_names and 'Frontal' in region_names:
        return coarse_positions
    return region_positions


def plot_region_connectivity(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    electrode_names: list = None,
    regions: dict = None,
    title: str = "Region Connectivity",
    save_path: str = None,
    ax: plt.Axes = None,
    cmap_name: str = 'hot',
    node_size: int = 2000,
    aggregation: str = 'mean',
    show_self_connections: bool = True,
    threshold_percentile: float = 0,
):
    """
    Visualize edge importance aggregated by brain region.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_mask: Edge importance values [num_edges]
        electrode_names: List of electrode names
        regions: Dictionary mapping region name to list of electrodes
        title: Plot title
        save_path: Path to save figure
        ax: Matplotlib axis
        cmap_name: Colormap name
        node_size: Size of region nodes
        aggregation: How to aggregate electrode edges ('mean', 'sum', 'max')
        show_self_connections: Whether to show within-region connections
        threshold_percentile: Only show connections above this percentile

    Returns:
        Matplotlib axis
    """
    from matplotlib.patches import FancyArrowPatch, Circle
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES
    if regions is None:
        regions = BRAIN_REGIONS_SIMPLE

    # Aggregate edge importance by region
    region_names, region_matrix, region_importance = aggregate_edge_importance_by_region(
        edge_index, edge_mask, electrode_names, regions, aggregation
    )

    # Get region positions
    region_pos = get_region_positions(regions)
    pos = np.array([region_pos[name] for name in region_names])
    num_regions = len(region_names)

    # Setup colormap
    cmap = plt.get_cmap(cmap_name)
    v_max = region_matrix.max()

    # Thresholding
    nonzero_vals = region_matrix[region_matrix > 0]
    if len(nonzero_vals) > 0 and threshold_percentile > 0:
        threshold = np.percentile(nonzero_vals, threshold_percentile)
    else:
        threshold = 0

    # Create figure if needed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        created_fig = True

    # Draw head outline
    head_circle = Circle((0, 0.3), 0.75, fill=False, color='black', linewidth=2)
    ax.add_patch(head_circle)
    # Nose
    ax.plot([0, 0.08, 0], [1.05, 1.15, 1.05], 'k-', linewidth=2)
    # Ears
    ax.plot([-0.75, -0.82, -0.75], [0.25, 0.3, 0.35], 'k-', linewidth=2)
    ax.plot([0.75, 0.82, 0.75], [0.25, 0.3, 0.35], 'k-', linewidth=2)

    # Draw connections between regions
    for i in range(num_regions):
        for j in range(i + 1, num_regions):  # Upper triangle only (undirected)
            w = region_matrix[i, j]
            if w < threshold:
                continue

            w_norm = w / (v_max + 1e-10)
            color = cmap(w_norm)
            linewidth = 1 + 8 * w_norm

            # Curved connection
            arrow = FancyArrowPatch(
                posA=pos[i], posB=pos[j],
                connectionstyle="arc3,rad=0.1",
                arrowstyle='-',
                color=color,
                linewidth=linewidth,
                alpha=0.5 + 0.5 * w_norm,
                zorder=2,
                shrinkA=np.sqrt(node_size) / 5,
                shrinkB=np.sqrt(node_size) / 5,
            )
            ax.add_patch(arrow)

    # Draw within-region connections (self-loops) as circles around nodes
    if show_self_connections:
        for i in range(num_regions):
            w = region_matrix[i, i]
            if w > threshold:
                w_norm = w / (v_max + 1e-10)
                self_circle = Circle(
                    pos[i],
                    radius=0.12,
                    fill=False,
                    color=cmap(w_norm),
                    linewidth=2 + 4 * w_norm,
                    alpha=0.5 + 0.5 * w_norm,
                    zorder=1,
                )
                ax.add_patch(self_circle)

    # Draw region nodes
    node_colors = [cmap(imp) for imp in region_importance]
    ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c=node_colors,
               edgecolors='black', linewidths=2, zorder=10)

    # Add region labels
    for i, name in enumerate(region_names):
        # Shorten labels for display
        short_name = name.replace('Frontal', 'F').replace('Central', 'C')
        short_name = short_name.replace('Parietal', 'P').replace('Temporal', 'T')
        short_name = short_name.replace('Occipital', 'O').replace('Prefrontal', 'PF')
        short_name = short_name.replace('-L', 'L').replace('-R', 'R').replace('-M', 'M')

        ax.text(pos[i, 0], pos[i, 1], short_name, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=11)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.6, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14)

    # Add colorbar
    if created_fig:
        norm = Normalize(vmin=0, vmax=v_max)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(f'Region Connectivity ({aggregation})', fontsize=10)

    if save_path and created_fig:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved region connectivity plot to {save_path}")

    if created_fig:
        plt.close()

    return ax


def plot_region_connectivity_subplots(
    edge_index: torch.Tensor,
    edge_masks: dict,
    class_names: list,
    electrode_names: list = None,
    regions: dict = None,
    title: str = "Region Connectivity by Class",
    save_path: str = None,
    cmap_name: str = 'hot',
    node_size: int = 1500,
    aggregation: str = 'mean',
    normalize_global: bool = True,
    ncols: int = None,
):
    """
    Visualize region-level connectivity for multiple classes.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_masks: Dictionary mapping class_idx/key -> edge_mask tensor
        class_names: List of class names
        electrode_names: List of electrode names
        regions: Dictionary mapping region name to list of electrodes
        title: Overall figure title
        save_path: Path to save figure
        cmap_name: Colormap name
        node_size: Size of region nodes
        aggregation: How to aggregate ('mean', 'sum', 'max')
        normalize_global: If True, use same color scale across all plots
        ncols: Number of columns

    Returns:
        Matplotlib figure
    """
    from matplotlib.patches import FancyArrowPatch, Circle
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES
    if regions is None:
        regions = BRAIN_REGIONS_SIMPLE

    num_plots = len(edge_masks)
    if ncols is None:
        ncols = min(num_plots, 3)
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))

    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    cmap = plt.get_cmap(cmap_name)

    # Pre-compute all region matrices
    all_region_data = {}
    global_max = 0

    for key, edge_mask in edge_masks.items():
        region_names, region_matrix, region_importance = aggregate_edge_importance_by_region(
            edge_index, edge_mask, electrode_names, regions, aggregation
        )
        all_region_data[key] = (region_names, region_matrix, region_importance)
        global_max = max(global_max, region_matrix.max())

    region_pos = get_region_positions(regions)

    # Plot each class
    for ax_idx, (key, (region_names, region_matrix, region_importance)) in enumerate(all_region_data.items()):
        ax = axes[ax_idx]
        label = class_names[ax_idx] if ax_idx < len(class_names) else str(key)

        pos = np.array([region_pos[name] for name in region_names])
        num_regions = len(region_names)

        v_max = global_max if normalize_global else region_matrix.max()

        # Draw head outline
        head_circle = Circle((0, 0.3), 0.75, fill=False, color='black', linewidth=2)
        ax.add_patch(head_circle)
        ax.plot([0, 0.08, 0], [1.05, 1.15, 1.05], 'k-', linewidth=2)
        ax.plot([-0.75, -0.82, -0.75], [0.25, 0.3, 0.35], 'k-', linewidth=2)
        ax.plot([0.75, 0.82, 0.75], [0.25, 0.3, 0.35], 'k-', linewidth=2)

        # Draw connections
        for i in range(num_regions):
            for j in range(i + 1, num_regions):
                w = region_matrix[i, j]
                if w < 1e-6:
                    continue

                w_norm = w / (v_max + 1e-10)
                color = cmap(w_norm)
                linewidth = 1 + 6 * w_norm

                arrow = FancyArrowPatch(
                    posA=pos[i], posB=pos[j],
                    connectionstyle="arc3,rad=0.1",
                    arrowstyle='-',
                    color=color,
                    linewidth=linewidth,
                    alpha=0.5 + 0.5 * w_norm,
                    zorder=2,
                    shrinkA=np.sqrt(node_size) / 5,
                    shrinkB=np.sqrt(node_size) / 5,
                )
                ax.add_patch(arrow)

        # Draw nodes
        node_colors = [cmap(imp) for imp in region_importance]
        ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c=node_colors,
                   edgecolors='black', linewidths=1.5, zorder=10)

        # Add labels
        for i, name in enumerate(region_names):
            short_name = name.replace('Frontal', 'F').replace('Central', 'C')
            short_name = short_name.replace('Parietal', 'P').replace('Temporal', 'T')
            short_name = short_name.replace('Occipital', 'O').replace('Prefrontal', 'PF')
            short_name = short_name.replace('-L', 'L').replace('-R', 'R').replace('-M', 'M')

            ax.text(pos[i, 0], pos[i, 1], short_name, ha='center', va='center',
                    fontsize=8, fontweight='bold', zorder=11)

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.6, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(label, fontsize=12)

    # Hide unused axes
    for ax_idx in range(num_plots, len(axes)):
        axes[ax_idx].set_visible(False)

    # Add shared colorbar
    norm = Normalize(vmin=0, vmax=global_max)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'Region Connectivity ({aggregation})', fontsize=10)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved region connectivity subplots to {save_path}")
    plt.close(fig)

    return fig


def plot_region_connectivity_matrix(
    edge_index: torch.Tensor,
    edge_masks: dict,
    class_names: list,
    electrode_names: list = None,
    regions: dict = None,
    title: str = "Region Connectivity Matrix",
    save_path: str = None,
    aggregation: str = 'mean',
    cmap_name: str = 'hot',
):
    """
    Plot region connectivity as heatmap matrices for comparison across classes.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_masks: Dictionary mapping class_idx/key -> edge_mask tensor
        class_names: List of class names
        electrode_names: List of electrode names
        regions: Dictionary mapping region name to list of electrodes
        title: Overall figure title
        save_path: Path to save figure
        aggregation: How to aggregate ('mean', 'sum', 'max')
        cmap_name: Colormap name

    Returns:
        Matplotlib figure
    """
    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES
    if regions is None:
        regions = BRAIN_REGIONS_SIMPLE

    num_plots = len(edge_masks)
    ncols = min(num_plots, 3)
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Pre-compute all region matrices to get global max
    all_matrices = {}
    global_max = 0
    region_names = None

    for key, edge_mask in edge_masks.items():
        names, matrix, _ = aggregate_edge_importance_by_region(
            edge_index, edge_mask, electrode_names, regions, aggregation
        )
        all_matrices[key] = matrix
        global_max = max(global_max, matrix.max())
        if region_names is None:
            region_names = names

    # Shorten region names for display
    short_names = []
    for name in region_names:
        short = name.replace('Frontal', 'F').replace('Central', 'C')
        short = short.replace('Parietal', 'P').replace('Temporal', 'T')
        short = short.replace('Occipital', 'O').replace('Prefrontal', 'PF')
        short = short.replace('-L', 'L').replace('-R', 'R').replace('-M', 'M')
        short_names.append(short)

    # Plot each class
    for ax_idx, (key, matrix) in enumerate(all_matrices.items()):
        ax = axes[ax_idx]
        label = class_names[ax_idx] if ax_idx < len(class_names) else str(key)

        im = ax.imshow(matrix, cmap=cmap_name, vmin=0, vmax=global_max, aspect='auto')

        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_title(label, fontsize=11)

    # Hide unused axes
    for ax_idx in range(num_plots, len(axes)):
        axes[ax_idx].set_visible(False)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'Connectivity ({aggregation})', fontsize=10)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved region connectivity matrix to {save_path}")
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
        title: Plot title
        save_path: Path to save figure
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
    N_SAMPLES_PER_CLASS = 1000
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

    # NEW: Arrow-based edge importance visualization (cleaner)
    print("\nGenerating arrow-based edge importance plots...")
    plot_edge_importance_arrows_subplots(
        edge_index=edge_index,
        edge_masks=edge_masks_dict,
        class_names=class_names,
        num_electrodes=62,
        top_k=N_LINES,
        title="Edge Importance by Class (Standard) - Arrow Plot",
        save_path=f"{OUTPUT_DIR}/edge_importance_arrows_standard.png",
        cmap_name='hot',
        node_size=300,
        display_labels=True,
        threshold_percentile=85,
        directed=False,
        normalize_global=True,
    )

    # Frequency band topomaps for ALL classes
    for idx, data in aggregated.items():
        node_mask = data['node_mask_mean']
        if node_mask.dim() == 2:
            plot_frequency_band_topomaps(
                node_mask,
                title=f"Frequency Band Importance - {class_names[idx]} (Standard)",
                save_path=f"{OUTPUT_DIR}/topomap_freq_bands_{class_names[idx].lower()}.png",
            )

    # NEW: Region-level connectivity visualization
    print("\nGenerating region-level connectivity plots...")

    # Region connectivity on head layout (simplified regions)
    plot_region_connectivity_subplots(
        edge_index=edge_index,
        edge_masks=edge_masks_dict,
        class_names=class_names,
        regions=BRAIN_REGIONS_SIMPLE,
        title="Region Connectivity by Class (Standard)",
        save_path=f"{OUTPUT_DIR}/region_connectivity_standard.png",
        cmap_name='hot',
        node_size=1500,
        aggregation='mean',
        normalize_global=True,
    )

    # Region connectivity matrix (heatmap view)
    plot_region_connectivity_matrix(
        edge_index=edge_index,
        edge_masks=edge_masks_dict,
        class_names=class_names,
        regions=BRAIN_REGIONS_SIMPLE,
        title="Region Connectivity Matrix (Standard)",
        save_path=f"{OUTPUT_DIR}/region_connectivity_matrix_standard.png",
        aggregation='mean',
        cmap_name='hot',
    )

    # Coarse regions (6 main areas) for even simpler view
    plot_region_connectivity_subplots(
        edge_index=edge_index,
        edge_masks=edge_masks_dict,
        class_names=class_names,
        regions=BRAIN_REGIONS_COARSE,
        title="Coarse Region Connectivity by Class (Standard)",
        save_path=f"{OUTPUT_DIR}/region_connectivity_coarse_standard.png",
        cmap_name='hot',
        node_size=2500,
        aggregation='mean',
        normalize_global=True,
    )

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
    # CLASS PROTOTYPE EXPLANATIONS (GNNExplainer Paper Method)
    # =========================================================================

    print("\n" + "="*70)
    print("CLASS PROTOTYPE EXPLANATIONS (GNNExplainer Paper Method)")
    print("="*70)
    print("\nThis follows the original GNNExplainer paper's approach for class-level explanations:")
    print("  1. Find reference instance (embedding closest to class mean)")
    print("  2. Compute explanations for many instances")
    print("  3. Align graphs (trivial for fixed EEG topology)")
    print("  4. Aggregate with MEDIAN for robustness")

    # Compute class prototypes for all classes
    N_PROTOTYPE_SAMPLES = min(100, N_SAMPLES_PER_CLASS)  # Use fewer samples for prototype demo

    prototypes = get_all_class_prototypes(
        model=explainer_model,
        explainer=explainer,
        data_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_classes=3,
        num_samples_per_class=N_PROTOTYPE_SAMPLES,
        use_contrastive=False,
    )

    # Visualize prototype explanations
    print("\nVisualizing class prototype explanations...")
    visualize_prototype_comparison(
        prototypes=prototypes,
        edge_index=edge_index,
        class_names=class_names,
        num_electrodes=62,
        n_lines=N_LINES,
        save_dir=OUTPUT_DIR,
    )

    # Visualize uncertainty in prototypes
    visualize_prototype_uncertainty(
        prototypes=prototypes,
        class_names=class_names,
        save_dir=OUTPUT_DIR,
    )

    # Visualize embedding space and reference selection
    visualize_embedding_space(
        prototypes=prototypes,
        class_names=class_names,
        save_dir=OUTPUT_DIR,
    )

    # Print prototype statistics
    print("\n" + "="*60)
    print("CLASS PROTOTYPE STATISTICS")
    print("="*60)
    for class_idx, proto in prototypes.items():
        class_name = class_names[class_idx]
        print(f"\n{class_name}:")
        print(f"  Reference instance index: {proto['reference_idx']}")
        print(f"  Distance to centroid: {proto['embedding_distances'][proto['reference_idx']]:.4f}")
        print(f"  Mean distance: {proto['embedding_distances'].mean():.4f}")
        print(f"  Samples used: {proto['num_samples']}")

        # Compare median vs mean
        median_node = proto['prototype_node_mask'].mean().item()
        mean_node = proto['mean_node_mask'].mean().item()
        print(f"  Avg node importance (median): {median_node:.4f}")
        print(f"  Avg node importance (mean): {mean_node:.4f}")

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

    # NEW: Arrow-based edge importance visualization for contrastive
    print("\nGenerating arrow-based edge importance plots (Contrastive)...")
    plot_edge_importance_arrows_subplots(
        edge_index=edge_index,
        edge_masks=contrastive_edge_masks,
        class_names=contrast_labels,
        num_electrodes=62,
        top_k=N_LINES,
        title="Contrastive Edge Importance - Arrow Plot",
        save_path=f"{OUTPUT_DIR}/edge_importance_arrows_contrastive.png",
        cmap_name='hot',
        node_size=250,
        display_labels=True,
        threshold_percentile=80,
        directed=False,
        normalize_global=False,  # Normalize per plot for contrastive
    )

    # Region-level connectivity for contrastive
    print("\nGenerating region-level connectivity plots (Contrastive)...")
    plot_region_connectivity_subplots(
        edge_index=edge_index,
        edge_masks=contrastive_edge_masks,
        class_names=contrast_labels,
        regions=BRAIN_REGIONS_SIMPLE,
        title="Region Connectivity (Contrastive)",
        save_path=f"{OUTPUT_DIR}/region_connectivity_contrastive.png",
        cmap_name='hot',
        node_size=1200,
        aggregation='mean',
        normalize_global=False,
    )

    plot_region_connectivity_matrix(
        edge_index=edge_index,
        edge_masks=contrastive_edge_masks,
        class_names=contrast_labels,
        regions=BRAIN_REGIONS_SIMPLE,
        title="Region Connectivity Matrix (Contrastive)",
        save_path=f"{OUTPUT_DIR}/region_connectivity_matrix_contrastive.png",
        aggregation='mean',
        cmap_name='hot',
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torcheeg.models import DGCNN


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================


class PyGGraphConvolution(MessagePassing):
    """Graph convolution using PyG's message passing for GNNExplainer compatibility."""

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
        # Transform features first
        x = torch.matmul(x, self.weight)

        # Propagate with edge weights
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

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

    For GNNExplainer compatibility, all convolutions use the SAME edge_index.
    We handle different Chebyshev polynomial orders by manipulating edge weights.
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
            edge_index: Edge indices [2, num_edges] - MUST include self-loops
            edge_weight: Edge weights [num_edges]
            num_nodes: Number of nodes
            self_loop_mask: Boolean mask indicating which edges are self-loops [num_edges]
        """
        result = None

        for k in range(self.num_layers):
            if k == 0:
                # T_0(L) * x = I * x
                # Use only self-loops: set non-self-loop weights to 0
                if self_loop_mask is not None:
                    weights_k = torch.where(
                        self_loop_mask,
                        torch.ones_like(edge_weight),
                        torch.zeros_like(edge_weight),
                    )
                else:
                    weights_k = edge_weight
            else:
                # T_1(L) * x = L * x (and higher orders)
                # Use the actual edge weights (excluding self-loops for pure L)
                if self_loop_mask is not None:
                    weights_k = torch.where(
                        self_loop_mask, torch.zeros_like(edge_weight), edge_weight
                    )
                else:
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
        """Compute normalized edge weights from learned adjacency."""
        A = F.relu(self.learned_A)

        # Get weights for the given edges
        edge_weight = A[edge_index[0], edge_index[1]]

        # Identify self-loops
        self_loop_mask = edge_index[0] == edge_index[1]

        # For normalization, we need to compute degrees from non-self-loop edges
        non_self_loop_mask = ~self_loop_mask

        row, col = edge_index
        deg = torch.zeros(self.num_electrodes, device=edge_index.device)

        # Only count non-self-loop edges for degree
        deg.scatter_add_(0, row[non_self_loop_mask], edge_weight[non_self_loop_mask])

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        # Normalize non-self-loop edges
        norm = torch.ones_like(edge_weight)
        norm[non_self_loop_mask] = (
            deg_inv_sqrt[row[non_self_loop_mask]]
            * edge_weight[non_self_loop_mask]
            * deg_inv_sqrt[col[non_self_loop_mask]]
        )

        # Self-loops keep weight 1
        norm[self_loop_mask] = 1.0

        return norm

    def get_edge_index_and_attr(self, threshold: float = 0.0):
        """
        Convert learned adjacency to edge format WITH self-loops included.
        This is crucial for GNNExplainer compatibility.
        """
        A = F.relu(self.learned_A)

        # Get non-self-loop edges
        if threshold > 0:
            mask = A > threshold
        else:
            mask = A > 1e-10

        # Remove diagonal from mask (we'll add self-loops separately)
        diag_mask = torch.eye(self.num_electrodes, dtype=torch.bool, device=A.device)
        mask = mask & ~diag_mask

        edge_index = torch.nonzero(mask, as_tuple=False).t().contiguous()
        edge_attr = A[edge_index[0], edge_index[1]]

        # Add self-loops
        self_loops = torch.arange(self.num_electrodes, device=A.device)
        self_loop_index = torch.stack([self_loops, self_loops])
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


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    from torch_geometric.explain import Explainer, GNNExplainer
    from torcheeg.datasets import SEEDDataset
    from torcheeg import transforms

    # Configuration
    THRESHOLD = 0.3
    BATCH_SIZE = 256
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    MODEL_PATH = "ckpts/dgcnn_seed_model.pth"

    ROOT_PATH = "/Users/urbansirca/datasets/SEED/Preprocessed_EEG"
    IO_PATH = "/Users/urbansirca/Desktop/FAX/Master's AI/MLGraphs/DGCNN/.torcheeg/datasets_1768912535105_zLBYu"

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
    visualize_learned_adjacency(model, threshold=THRESHOLD, save_path="learned_adjacency.png")

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

    # # Get samples from each class
    # print("\nGenerating Explanations...")
    # class_samples = {0: None, 1: None, 2: None}
    # class_names = ["Negative", "Neutral", "Positive"]

    # for x, y in test_loader:
    #     for i in range(len(y)):
    #         label = y[i].item()
    #         if class_samples[label] is None:
    #             class_samples[label] = x[i]
    #         if all(v is not None for v in class_samples.values()):
    #             break
    #     if all(v is not None for v in class_samples.values()):
    #         break

    # # Generate explanations
    # for class_idx, sample in class_samples.items():
    #     if sample is None:
    #         continue

    #     print(f"\nExplaining sample from class: {class_names[class_idx]}")

    #     sample_for_explainer = sample.squeeze()
    #     print(f"  Sample shape: {sample_for_explainer.shape}")

    #     explanation = explainer(
    #         x=sample_for_explainer,
    #         edge_index=edge_index,
    #         edge_weight=edge_weight,
    #     )

    #     print(f"  Node mask shape: {explanation.node_mask.shape}")
    #     print(f"  Edge mask shape: {explanation.edge_mask.shape}")

    #     # Visualize
    #     node_mask = explanation.node_mask.squeeze()
    #     visualize_node_importance(
    #         node_mask,
    #         title=f"Node Importance - {class_names[class_idx]} Emotion",
    #         save_path=f"node_importance_class_{class_idx}.png",
    #     )

    #     visualize_edge_importance(
    #         edge_index,
    #         explanation.edge_mask,
    #         num_electrodes=62,
    #         top_k=50,
    #         title=f"Edge Importance - {class_names[class_idx]} Emotion",
    #         save_path=f"edge_importance_class_{class_idx}.png",
    #     )

    #     # Visualize edges - standard plot
    #     visualize_edge_importance(
    #         edge_index,
    #         explanation.edge_mask,
    #         num_electrodes=62,
    #         top_k=50,
    #         title=f"Edge Importance - {class_names[class_idx]} Emotion",
    #         save_path=f"edge_importance_class_{class_idx}.png",
    #     )

    #     # Visualize edges - circular plot
    #     visualize_edge_importance_circular(
    #         edge_index,
    #         explanation.edge_mask,
    #         num_electrodes=62,
    #         n_lines=50,
    #         title=f"Edge Importance - {class_names[class_idx]} Emotion",
    #         save_path=f"edge_importance_circular_class_{class_idx}.png",
    #     )

    # print("\nDone!")


    print("\nGenerating aggregated explanations...")
    aggregated = get_aggregated_explanations(
        explainer=explainer,
        data_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_samples_per_class=50,  # Adjust based on time constraints
        num_classes=3,
    )
    
    class_names = ["Negative", "Neutral", "Positive"]
    
    # Visualize aggregated results
    for class_idx, data in aggregated.items():
        print(f"\nClass: {class_names[class_idx]} (n={data['num_samples']})")
        
        # Node importance (averaged)
        visualize_node_importance(
            data['node_mask_mean'],
            title=f"Avg Node Importance - {class_names[class_idx]} (n={data['num_samples']})",
            save_path=f"avg_node_importance_class_{class_idx}.png",
        )
        
        # Edge importance (averaged)
        visualize_edge_importance_circular(
            edge_index,
            data['edge_mask_mean'],
            num_electrodes=62,
            n_lines=50,
            title=f"Avg Edge Importance - {class_names[class_idx]}",
            save_path=f"avg_edge_importance_circular_class_{class_idx}.png",
        )

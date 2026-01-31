"""
Visualization utilities for DGCNN EEG analysis.

Contains functions for:
- Node/edge importance visualization
- MNE topographic plots
- Region-level connectivity
- Validation metric plots
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import mne
import os

# Frequency band names
FREQUENCY_BANDS = ['Delta (1-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-14 Hz)', 'Beta (14-31 Hz)', 'Gamma (31-49 Hz)']

SEED_ELECTRODE_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]

# Brain region groupings
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


# ============================================================================
# ELECTRODE POSITIONS
# ============================================================================

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


# ============================================================================
# MNE ELECTRODE POSITIONS & TOPOGRAPHIC VISUALIZATION
# ============================================================================

def get_mne_montage():
    """Create an MNE montage for the 62-channel SEED electrode layout."""
    standard_montage = mne.channels.make_standard_montage('standard_1020')
    standard_pos = standard_montage.get_positions()['ch_pos']

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
            o1_pos = standard_pos.get('O1', np.array([-0.03, -0.1, 0]))
            positions[seed_name] = o1_pos + np.array([-0.02, -0.03, -0.02])
        elif seed_name == 'CB2':
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


# ============================================================================
# NODE IMPORTANCE VISUALIZATION
# ============================================================================

# def visualize_node_importance(
#     node_mask: torch.Tensor,
#     electrode_names: list = None,
#     title: str = "Node Importance",
#     save_path: str = None,
# ):
#     """Visualize node (electrode) importance on a head layout."""
#     if electrode_names is None:
#         electrode_names = SEED_ELECTRODE_NAMES

#     num_electrodes = len(node_mask)
#     positions = get_electrode_positions(num_electrodes)

#     if node_mask.dim() > 1:
#         importance = node_mask.mean(dim=-1).cpu().numpy()
#     else:
#         importance = node_mask.cpu().numpy()

#     importance = (importance - importance.min()) / (
#         importance.max() - importance.min() + 1e-10
#     )

#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))

#     circle = plt.Circle((0, 0.35), 0.65, fill=False, color="black", linewidth=2)
#     ax.add_patch(circle)
#     ax.plot([0, 0.1, 0], [1.0, 1.1, 1.0], "k-", linewidth=2)
#     ax.plot([-0.65, -0.7, -0.65], [0.3, 0.35, 0.4], "k-", linewidth=2)
#     ax.plot([0.65, 0.7, 0.65], [0.3, 0.35, 0.4], "k-", linewidth=2)

#     scatter = ax.scatter(
#         positions[:, 0],
#         positions[:, 1],
#         c=importance,
#         cmap="Reds",
#         s=500,
#         edgecolors="black",
#         linewidths=1,
#         vmin=0,
#         vmax=1,
#     )

#     for i, (x, y) in enumerate(positions):
#         ax.annotate(
#             electrode_names[i] if i < len(electrode_names) else str(i),
#             (x, y),
#             ha="center",
#             va="center",
#             fontsize=6,
#             fontweight="bold",
#         )

#     plt.colorbar(scatter, ax=ax, label="Importance")
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-0.5, 1.2)
#     ax.set_aspect("equal")
#     ax.axis("off")
#     ax.set_title(title, fontsize=14)

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.show()

#     return fig


def visualize_node_importance_subplots(
    node_masks: dict,
    class_names: list,
    electrode_names: list = None,
    title: str = "Node Importance by Class",
    save_path: str = None,
    ncols: int = None,
    normalize_per_plot: bool = False,
):
    """Visualize node importance for multiple classes as subplots."""
    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES

    num_plots = len(node_masks)

    if ncols is None:
        if num_plots <= 3:
            ncols = num_plots
        else:
            ncols = 3
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    first_mask = list(node_masks.values())[0]
    num_electrodes = len(first_mask) if first_mask.dim() == 1 else first_mask.shape[0]
    positions = get_electrode_positions(num_electrodes)

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

        if normalize_per_plot:
            local_min, local_max = importance.min(), importance.max()
            importance_norm = (importance - local_min) / (local_max - local_min + 1e-10)
        else:
            importance_norm = (importance - global_min) / (global_max - global_min + 1e-10)

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

        label = class_names[ax_idx] if ax_idx < len(class_names) else str(key)
        ax.set_title(label, fontsize=10)

    for ax_idx in range(num_plots, len(axes)):
        axes[ax_idx].set_visible(False)

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
# EDGE IMPORTANCE VISUALIZATION
# ============================================================================

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

    non_self_loop = edge_idx_np[0] != edge_idx_np[1]
    edge_importance_filtered = edge_importance[non_self_loop]
    edge_idx_filtered = edge_idx_np[:, non_self_loop]

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

    conn_matrix = np.zeros((num_electrodes, num_electrodes))
    for i in range(edge_idx_np.shape[1]):
        src, dst = edge_idx_np[0, i], edge_idx_np[1, i]
        if src != dst:
            conn_matrix[src, dst] = edge_importance[i]

    conn_matrix = (conn_matrix + conn_matrix.T) / 2

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
    """Visualize edge importance for multiple classes as subplots."""
    from mne.viz import circular_layout
    from mne_connectivity.viz import plot_connectivity_circle

    if electrode_names is None:
        electrode_names = SEED_ELECTRODE_NAMES[:num_electrodes]

    edge_idx_np = edge_index.cpu().numpy()

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

    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

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
        label = class_names[ax_idx] if ax_idx < len(class_names) else str(key)
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

    for ax_idx in range(num_plots, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(title, fontsize=14, color="white", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.show()

    return fig


# ============================================================================
# MNE TOPOMAP VISUALIZATION
# ============================================================================


def plot_topomap_node_importance_subplots(
    node_masks: dict,
    class_names: list,
    title: str = "Node Importance - MNE Topomap",
    save_path: str = None,
    normalize_global: bool = True,
):
    """Plot node importance topomaps for multiple classes as subplots."""
    num_plots = len(node_masks)
    ncols = min(num_plots, 3)
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

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


def plot_frequency_band_topomaps(
    node_mask,
    title: str = "Frequency Band Importance",
    save_path: str = None,
):
    """Plot separate topomaps for each frequency band."""
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
# ADJACENCY MATRIX VISUALIZATION
# ============================================================================

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



# ============================================================================
# PROTOTYPE VISUALIZATION
# ============================================================================

# def visualize_prototype_comparison(
#     prototypes: dict,
#     edge_index: torch.Tensor,
#     class_names: list,
#     num_electrodes: int = 62,
#     n_lines: int = 50,
#     save_dir: str = "plots",
# ):
#     """Visualize class prototype explanations (median aggregation only)."""
#     os.makedirs(save_dir, exist_ok=True)

#     median_node_masks = {idx: p['prototype_node_mask'] for idx, p in prototypes.items()}
#     median_edge_masks = {idx: p['prototype_edge_mask'] for idx, p in prototypes.items()}

#     # Node importance topomap (median)
#     plot_topomap_node_importance_subplots(
#         node_masks=median_node_masks,
#         class_names=class_names,
#         title="Class Prototype Node Importance (Median)",
#         save_path=f"{save_dir}/prototype_topomap_node.png",
#     )

#     # Edge importance circular plot (median)
#     visualize_edge_importance_circular_subplots(
#         edge_index=edge_index,
#         edge_masks=median_edge_masks,
#         class_names=class_names,
#         num_electrodes=num_electrodes,
#         n_lines=n_lines,
#         title="Class Prototype Edge Importance (Median)",
#         save_path=f"{save_dir}/prototype_edge.png",
#     )

#     print(f"\nPrototype visualizations saved to {save_dir}/")


# def visualize_prototype_uncertainty(
#     prototypes: dict,
#     class_names: list,
#     save_dir: str = "plots",
# ):
#     """Visualize uncertainty/variance in prototype explanations."""
#     os.makedirs(save_dir, exist_ok=True)

#     num_classes = len(prototypes)
#     fig, axes = plt.subplots(2, num_classes, figsize=(5 * num_classes, 10))

#     for idx, (class_idx, proto) in enumerate(prototypes.items()):
#         class_name = class_names[idx] if idx < len(class_names) else str(class_idx)

#         ax_node = axes[0, idx] if num_classes > 1 else axes[0]
#         node_std = proto['std_node_mask'].mean(dim=-1).numpy()

#         positions = get_electrode_positions(62)
#         scatter = ax_node.scatter(
#             positions[:, 0], positions[:, 1],
#             c=node_std, cmap='Blues', s=200,
#             edgecolors='black', linewidths=1,
#         )
#         ax_node.set_title(f"{class_name}\nNode Importance Std Dev")
#         ax_node.set_aspect('equal')
#         ax_node.axis('off')
#         plt.colorbar(scatter, ax=ax_node, shrink=0.7)

#         ax_edge = axes[1, idx] if num_classes > 1 else axes[1]
#         edge_std = proto['std_edge_mask'].numpy()
#         ax_edge.hist(edge_std, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
#         ax_edge.set_xlabel('Edge Importance Std Dev')
#         ax_edge.set_ylabel('Count')
#         ax_edge.set_title(f"{class_name}\nEdge Importance Variance Distribution")

#     plt.suptitle("Prototype Explanation Uncertainty", fontsize=14, y=1.02)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/prototype_uncertainty.png", dpi=150, bbox_inches='tight')
#     plt.close(fig)

#     print(f"Uncertainty visualization saved to {save_dir}/prototype_uncertainty.png")


# def visualize_embedding_space(
#     prototypes: dict,
#     class_names: list,
#     save_dir: str = "plots",
# ):
#     """Visualize the embedding space and reference instance selection."""
#     os.makedirs(save_dir, exist_ok=True)

#     fig, axes = plt.subplots(1, len(prototypes), figsize=(5 * len(prototypes), 5))
#     if len(prototypes) == 1:
#         axes = [axes]

#     for idx, (class_idx, proto) in enumerate(prototypes.items()):
#         ax = axes[idx]
#         class_name = class_names[idx] if idx < len(class_names) else str(class_idx)

#         distances = proto['embedding_distances'].numpy()
#         ref_idx = proto['reference_idx']

#         ax.hist(distances, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
#         ax.axvline(distances[ref_idx], color='red', linestyle='--', linewidth=2,
#                    label=f'Reference (idx={ref_idx})')
#         ax.axvline(distances.mean(), color='green', linestyle=':', linewidth=2,
#                    label=f'Mean dist={distances.mean():.3f}')
#         ax.set_xlabel('Distance to Class Centroid')
#         ax.set_ylabel('Count')
#         ax.set_title(f"{class_name}\nEmbedding Distances")
#         ax.legend()

#     plt.suptitle("Reference Instance Selection: Distance to Class Centroid", fontsize=14, y=1.02)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/prototype_embedding_distances.png", dpi=150, bbox_inches='tight')
#     plt.close(fig)

#     print(f"Embedding distance visualization saved to {save_dir}/prototype_embedding_distances.png")


# ============================================================================
# VALIDATION METRIC PLOTS
# ============================================================================

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

"""
Plot electrode importance on brain topography using MNE.
Uses the SEED dataset's 62-channel electrode layout.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne

# SEED dataset uses 62 electrodes with the following standard 10-20 names
# Reference: SEED dataset documentation
SEED_CHANNEL_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
    'O2', 'CB2'
]

# Create MNE montage (standard 10-20 positions)
# Some SEED channels may not be in standard montage, we'll handle that
def get_montage():
    """Get electrode positions, using standard 10-20 where possible."""
    montage = mne.channels.make_standard_montage('standard_1020')
    return montage


def plot_topomap(values, title="Electrode Importance", save_path=None, cmap='RdBu_r'):
    """
    Plot electrode values on a topographic brain map.

    Args:
        values: Array of shape [62] with importance values per electrode
        title: Plot title
        save_path: Path to save the figure (optional)
        cmap: Colormap to use
    """
    # Get standard montage and create case-insensitive lookup
    montage = mne.channels.make_standard_montage('standard_1020')
    # Create mapping from uppercase to actual montage name
    montage_lookup = {ch.upper(): ch for ch in montage.ch_names}

    # Map SEED channels to standard montage names
    valid_channels = []
    valid_values = []
    missing_channels = []

    for i, ch in enumerate(SEED_CHANNEL_NAMES):
        ch_upper = ch.upper()

        # Skip cerebellum electrodes (not in standard 10-20)
        if ch_upper in ['CB1', 'CB2']:
            missing_channels.append(ch)
            continue

        # Look up the proper case version
        if ch_upper in montage_lookup:
            valid_channels.append(montage_lookup[ch_upper])
            valid_values.append(values[i])
        else:
            missing_channels.append(ch)

    if missing_channels:
        print(f"Note: {len(missing_channels)} channels not in standard montage: {missing_channels}")

    # Create info object with proper channel names
    info = mne.create_info(ch_names=valid_channels, sfreq=1, ch_types='eeg')
    info.set_montage('standard_1020', on_missing='ignore')

    # Create evoked object for plotting
    valid_values = np.array(valid_values)
    evoked = mne.EvokedArray(valid_values.reshape(-1, 1), info)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    mne.viz.plot_topomap(
        evoked.data[:, 0],
        evoked.info,
        axes=ax,
        cmap=cmap,
        show=False,
        contours=6,
        sensors=True
    )
    ax.set_title(title, fontsize=14)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=valid_values.min(), vmax=valid_values.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Importance')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_connectivity(edge_mask, title="Electrode Connectivity", save_path=None, top_k=50):
    """
    Plot electrode connectivity on brain topography.

    Args:
        edge_mask: Array of shape [62, 62] with edge importance
        title: Plot title
        save_path: Path to save the figure
        top_k: Number of top edges to display
    """
    from mne.viz import circular_layout
    from mne_connectivity.viz import plot_connectivity_circle

    # Get top-k edges
    flat = edge_mask.flatten()
    threshold = np.sort(flat)[-top_k]

    # Create connectivity matrix (thresholded)
    conn = np.where(edge_mask >= threshold, edge_mask, 0)

    # Get node order for circular layout (group by brain region)
    # Frontal, Central, Parietal, Temporal, Occipital
    regions = {
        'Frontal': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
        'Frontocentral': ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
        'Central': ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
        'Centroparietal': ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
        'Parietal': ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
        'Occipital': ['PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    }

    # Create ordered list and colors
    node_order = []
    node_colors = []
    colors = plt.cm.Set2(np.linspace(0, 1, len(regions)))

    for idx, (region, channels) in enumerate(regions.items()):
        for ch in channels:
            if ch in SEED_CHANNEL_NAMES:
                node_order.append(SEED_CHANNEL_NAMES.index(ch))
                node_colors.append(colors[idx])

    # Reorder connectivity matrix
    conn_ordered = conn[np.ix_(node_order, node_order)]
    labels_ordered = [SEED_CHANNEL_NAMES[i] for i in node_order]

    # Plot connectivity circle
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    plot_connectivity_circle(
        conn_ordered,
        labels_ordered,
        n_lines=top_k,
        node_angles=circular_layout(labels_ordered, labels_ordered, start_pos=90),
        node_colors=node_colors,
        title=title,
        fig=fig,
        show=False,
        colormap='hot',
        vmin=0,
        vmax=conn.max()
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def compute_node_importance_from_edges(edge_mask):
    """
    Derive node importance from edge importance.

    For each node, importance = sum of edge weights connected to it.
    This is more meaningful than learned node masks for graph-level tasks.
    """
    # Sum incoming and outgoing edge weights for each node
    # edge_mask is [62, 62], edge_mask[i,j] = importance of edge from i to j
    importance = edge_mask.sum(axis=0) + edge_mask.sum(axis=1)

    # Normalize to [0, 1]
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    return importance


def main():
    import os

    OUTPUT_DIR = "explanations_contrastive"
    CLASS_NAMES = ['negative', 'neutral', 'positive']

    for class_name in CLASS_NAMES:
        # Load edge mask (node mask is nearly uniform, so we derive node importance from edges)
        edge_mask_path = os.path.join(OUTPUT_DIR, f'mean_edge_mask_{class_name}.npy')

        if not os.path.exists(edge_mask_path):
            print(f"Skipping {class_name}: files not found")
            continue

        # Load data
        edge_mask = np.load(edge_mask_path)  # [62, 62]

        # Derive node importance from edge connectivity
        # This is more meaningful than learned node masks for DGCNN
        node_importance = compute_node_importance_from_edges(edge_mask)  # [62]

        print(f"\n{'='*50}")
        print(f"CLASS: {class_name.upper()}")
        print(f"{'='*50}")

        # Plot topomap of electrode importance
        print(f"\nPlotting electrode importance topomap for {class_name}...")
        plot_topomap(
            node_importance,
            title=f"Electrode Importance - {class_name.capitalize()}",
            save_path=os.path.join(OUTPUT_DIR, f'topomap_{class_name}.png')
        )

        # Skip per-frequency-band plot since we're using edge-derived importance
        # (Node masks are uniform, so per-band analysis isn't meaningful)
        # Instead, show the single aggregate importance map

        # Plot connectivity
        print(f"\nPlotting connectivity for {class_name}...")
        try:
            plot_connectivity(
                edge_mask,
                title=f"Top-50 Connections - {class_name.capitalize()}",
                save_path=os.path.join(OUTPUT_DIR, f'connectivity_{class_name}.png'),
                top_k=50
            )
        except Exception as e:
            print(f"Connectivity plot failed: {e}")

        # Print top electrodes
        top_indices = np.argsort(node_importance)[-10:][::-1]
        print(f"\nTop 10 electrodes for {class_name}:")
        for idx in top_indices:
            print(f"  {SEED_CHANNEL_NAMES[idx]}: {node_importance[idx]:.4f}")


if __name__ == '__main__':
    main()

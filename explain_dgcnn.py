"""explain_dgcnn.py

Script to explain DGCNN model predictions using the GNN Explainer.
Supports per-class analysis and aggregation over multiple samples.
"""

import os
import torch
import numpy as np
from collections import defaultdict
from torcheeg.datasets import SEEDDataset
from torcheeg import transforms
from torcheeg.models import DGCNN
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle

from explainer.explain import Explainer


# Define electrode names for 62-channel SEED dataset
ELECTRODE_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
    'O2', 'CB2'
]

# Frequency band names (matching the order in BandDifferentialEntropy)
FREQUENCY_BANDS = ['Delta (1-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-14 Hz)', 'Beta (14-31 Hz)', 'Gamma (31-49 Hz)']
BAND_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def get_electrode_layout():
    """Get electrode layout for circular visualization."""
    # Group electrodes by region
    lh_electrodes = [name for name in ELECTRODE_NAMES if name.endswith('1') or
                     name.endswith('3') or name.endswith('5') or name.endswith('7')]
    mid_electrodes = [name for name in ELECTRODE_NAMES if name.endswith('Z')]
    rh_electrodes = [name for name in ELECTRODE_NAMES if name.endswith('2') or
                     name.endswith('4') or name.endswith('6') or name.endswith('8')]

    # Create node order: left -> midline -> right
    node_order = lh_electrodes + mid_electrodes + rh_electrodes

    # Create circular layout
    node_angles = circular_layout(
        ELECTRODE_NAMES,
        node_order,
        start_pos=90,
        group_boundaries=[0, len(lh_electrodes), len(lh_electrodes) + len(mid_electrodes)]
    )

    # Assign colors by region
    node_colors = []
    for name in ELECTRODE_NAMES:
        if name in lh_electrodes:
            node_colors.append('steelblue')
        elif name in mid_electrodes:
            node_colors.append('green')
        else:
            node_colors.append('crimson')

    return node_angles, node_colors


class Args:
    """Configuration class with all parameters."""
    # Model parameters
    model_path = "ckpts/dgcnn_seed_model.pth"
    in_channels = 5
    num_electrodes = 62
    hid_channels = 32
    num_layers = 2
    num_classes = 3

    # Explainer parameters
    num_epochs = 100
    num_gc_layers = 1
    mask_act = 'sigmoid'
    mask_bias = False

    # Naming parameters (used by io_utils.gen_prefix)
    bmname = None
    method = "base"
    hidden_dim = 32
    output_dim = 32
    bias = True
    name_suffix = ""
    explainer_suffix = ""

    # Optimizer parameters
    lr = 0.01
    opt = 'adam'
    opt_scheduler = "cos"
    opt_decay_step = 20
    opt_decay_rate = 0.8
    opt_restart = 50

    # Data parameters
    graph_idx = 0
    num_samples = 5

    # Output parameters
    logdir = 'log/explain'
    dataset = 'seed'

    # Device
    gpu = False

    # Explanation parameters
    samples_per_class = 20          # Number of samples to explain per class
    correct_only = True             # Only use correctly predicted samples
    top_k = 10                      # Number of top edges to show
    class_filter = None             # List of classes to analyze (None = all), e.g. [0, 2]
    n_lines = 10                    # Number of top connections in circular plot
    
    contrastive = True              # Whether to use contrastive explanations
    contrastive_class = 0            # Class to contrast against if not using predicted class


def load_model(args, device):
    """Load the trained DGCNN model."""
    model = DGCNN(
        in_channels=args.in_channels,
        num_electrodes=args.num_electrodes,
        hid_channels=args.hid_channels,
        num_layers=args.num_layers,
        num_classes=args.num_classes
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_data():
    """Load the SEED dataset."""
    dataset = SEEDDataset(
        root_path='/Users/urbansirca/datasets/SEED/Preprocessed_EEG',
        io_path='/Users/urbansirca/Desktop/FAX/Master\'s AI/MLGraphs/DGCNN/.torcheeg/datasets_1768912535105_zLBYu',
        offline_transform=transforms.BandDifferentialEntropy(band_dict={
            "delta": [1, 4],
            "theta": [4, 8],
            "alpha": [8, 14],
            "beta": [14, 31],
            "gamma": [31, 49]
        }),
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Compose([
            transforms.Select('emotion'),
            transforms.Lambda(lambda x: x + 1)
        ])
    )
    return dataset


def get_samples_by_class(dataset, model, device, samples_per_class=10, correct_only=True):
    """Get samples organized by class label."""
    class_samples = defaultdict(list)
    class_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    print(f"Scanning dataset for samples (correct_only={correct_only})...")

    for i in range(len(dataset)):
        # Check if we have enough for all 3 classes
        if all(len(class_samples[c]) >= samples_per_class for c in [0, 1, 2]):
            break

        x, y = dataset[i]
        x_tensor = x.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_tensor)
            pred_class = pred.argmax(dim=1).item()
            pred_probs = torch.softmax(pred, dim=1).cpu().numpy()

        # Only add if we need more samples for this class
        if len(class_samples[y]) < samples_per_class:
            if correct_only and pred_class != y:
                continue
            class_samples[y].append({
                'index': i,
                'features': x.numpy(),
                'label': y,
                'pred_class': pred_class,
                'pred_probs': pred_probs[0]
            })

        if i % 1000 == 0:
            counts = {class_names.get(k, k): len(v) for k, v in class_samples.items()}
            print(f"  Scanned {i} samples, found: {counts}")

    # Final count
    counts = {class_names.get(k, k): len(v) for k, v in class_samples.items()}
    print(f"  Final: {counts}")

    return class_samples


def run_explanations_for_class(model, samples, args, device, class_name):
    """Run explainer for a list of samples from one class."""
    if not samples:
        return None, None

    print(f"\n{'='*60}")
    print(f"Running explanations for class: {class_name} ({len(samples)} samples)")
    print(f"{'='*60}")

    # Prepare data
    features_list = [s['features'] for s in samples]
    labels_list = [s['label'] for s in samples]
    preds_list = [s['pred_probs'] for s in samples]

    feat = np.stack(features_list, axis=0)
    labels = np.array(labels_list)
    pred = np.stack(preds_list, axis=0)

    # Get adjacency from the model
    adj = model.A.detach().cpu().numpy()
    adj = np.tile(adj, (len(samples), 1, 1))

    # Create explainer
    writer = SummaryWriter(os.path.join(args.logdir, class_name.lower()))

    explainer = Explainer(
        model=model,
        adj=adj,
        feat=feat,
        label=labels,
        pred=pred,
        train_idx=np.arange(len(samples)),
        args=args,
        writer=writer,
        print_training=False,  # Reduce output noise
        graph_mode=True,
        graph_idx=0
    )

    # Run explanations
    graph_indices = list(range(len(samples)))
    masked_adjs, feat_masks = explainer.explain_graphs(graph_indices, contrastive=args.contrastive)

    writer.close()

    return masked_adjs, feat_masks, samples


def analyze_class_explanations(masked_adjs, feat_masks, samples, class_name, top_k=10):
    """Analyze and aggregate explanations for a class."""
    if masked_adjs is None:
        return None

    print(f"\n--- {class_name} Class Analysis ---")

    # Aggregate edge importance across all samples
    all_adjs = np.stack(masked_adjs, axis=0)
    mean_adj = np.mean(all_adjs, axis=0)
    std_adj = np.std(all_adjs, axis=0)

    # Aggregate feature importance across all samples
    all_feat_masks = np.stack(feat_masks, axis=0)
    mean_feat = np.mean(all_feat_masks, axis=0)
    std_feat = np.std(all_feat_masks, axis=0)

    # Get top edges by mean importance
    flat_idx = np.argsort(mean_adj.flatten())[-top_k:][::-1]
    top_edges = np.unravel_index(flat_idx, mean_adj.shape)

    print(f"\nTop {top_k} most important edges (averaged over {len(samples)} samples):")
    edge_data = []
    for j in range(top_k):
        src, dst = top_edges[0][j], top_edges[1][j]
        mean_weight = mean_adj[src, dst]
        std_weight = std_adj[src, dst]
        print(f"  Electrode {src:2d} <-> Electrode {dst:2d}: {mean_weight:.4f} ± {std_weight:.4f}")
        edge_data.append((src, dst, mean_weight, std_weight))

    # Print feature (frequency band) importance
    print(f"\nFrequency Band Importance (averaged over {len(samples)} samples):")
    sorted_bands = np.argsort(mean_feat)[::-1]
    for idx in sorted_bands:
        print(f"  {FREQUENCY_BANDS[idx]}: {mean_feat[idx]:.4f} ± {std_feat[idx]:.4f}")

    # Print per-sample summary
    print(f"\nPer-sample prediction accuracy:")
    correct = sum(1 for s in samples if s['pred_class'] == s['label'])
    print(f"  {correct}/{len(samples)} correctly predicted")

    return {
        'mean_adj': mean_adj,
        'std_adj': std_adj,
        'top_edges': edge_data,
        'mean_feat': mean_feat,
        'std_feat': std_feat,
        'samples': samples,
        'masked_adjs': masked_adjs,
        'feat_masks': feat_masks
    }


def plot_aggregated_results(results, save_path):
    """Plot aggregated importance matrices for each class."""
    class_names = ['Negative', 'Neutral', 'Positive']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, class_name in enumerate(class_names):
        if class_name not in results or results[class_name] is None:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(f'{class_name} (Class {idx})')
            continue

        mean_adj = results[class_name]['mean_adj']
        im = axes[idx].imshow(mean_adj, cmap='hot', aspect='auto')
        axes[idx].set_title(f'{class_name} (Class {idx})')
        axes[idx].set_xlabel('Electrode')
        axes[idx].set_ylabel('Electrode')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved aggregated plot to {save_path}")


def plot_top_edges_comparison(results, save_path, top_k=10):
    """Plot comparison of top edges across classes."""
    class_names = ['Negative', 'Neutral', 'Positive']
    fig, ax = plt.subplots(figsize=(12, 6))

    x_positions = np.arange(top_k)
    width = 0.25
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for idx, class_name in enumerate(class_names):
        if class_name not in results or results[class_name] is None:
            continue

        top_edges = results[class_name]['top_edges']
        weights = [e[2] for e in top_edges[:top_k]]
        stds = [e[3] for e in top_edges[:top_k]]

        offset = (idx - 1) * width
        bars = ax.bar(x_positions + offset, weights, width, label=class_name,
                      color=colors[idx], yerr=stds, capsize=3, alpha=0.8)

    ax.set_xlabel('Edge Rank')
    ax.set_ylabel('Mean Importance')
    ax.set_title('Top Edge Importance by Class')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'#{i+1}' for i in range(top_k)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved edge comparison plot to {save_path}")


def plot_connectivity_circles(results, save_dir, n_lines=30):
    """Plot circular connectivity diagrams for each class."""
    class_names = ['Negative', 'Neutral', 'Positive']
    node_angles, node_colors = get_electrode_layout()

    for class_name in class_names:
        if class_name not in results or results[class_name] is None:
            continue

        mean_adj = results[class_name]['mean_adj']

        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black', subplot_kw=dict(polar=True))
        plot_connectivity_circle(
            mean_adj,
            ELECTRODE_NAMES,
            n_lines=n_lines,
            node_angles=node_angles,
            node_colors=node_colors,
            title=f'{class_name} - Important Connections',
            ax=ax,
            colormap='hot',
            fontsize_names=8,
            fontsize_title=14,
            padding=2.0,
            show=False
        )

        save_path = os.path.join(save_dir, f'connectivity_circle_{class_name.lower()}.png')
        fig.savefig(save_path, dpi=150, facecolor='black')
        plt.close(fig)
        print(f"Saved circular connectivity plot to {save_path}")

    # Also create a combined figure with all three classes side by side
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), facecolor='black',
                              subplot_kw=dict(polar=True))

    for idx, class_name in enumerate(class_names):
        if class_name not in results or results[class_name] is None:
            axes[idx].set_facecolor('black')
            axes[idx].set_title(f'{class_name} - No data', color='white', fontsize=14)
            continue

        mean_adj = results[class_name]['mean_adj']
        plot_connectivity_circle(
            mean_adj,
            ELECTRODE_NAMES,
            n_lines=n_lines,
            node_angles=node_angles,
            node_colors=node_colors,
            title=f'{class_name}',
            ax=axes[idx],
            colormap='hot',
            fontsize_names=7,
            fontsize_title=14,
            padding=2.0,
            show=False
        )

    save_path = os.path.join(save_dir, 'connectivity_circles_all_classes.png')
    fig.savefig(save_path, dpi=150, facecolor='black')
    plt.close(fig)
    print(f"Saved combined circular connectivity plot to {save_path}")


def plot_feature_importance(results, save_path):
    """Plot frequency band importance comparison across classes."""
    class_names = ['Negative', 'Neutral', 'Positive']
    class_colors = ['#e74c3c', '#3498db', '#2ecc71']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Grouped bar chart comparing classes
    ax1 = axes[0]
    x = np.arange(len(FREQUENCY_BANDS))
    width = 0.25

    for idx, class_name in enumerate(class_names):
        if class_name not in results or results[class_name] is None:
            continue
        mean_feat = results[class_name]['mean_feat']
        std_feat = results[class_name]['std_feat']
        offset = (idx - 1) * width
        ax1.bar(x + offset, mean_feat, width, label=class_name,
                color=class_colors[idx], yerr=std_feat, capsize=3, alpha=0.8)

    ax1.set_xlabel('Frequency Band')
    ax1.set_ylabel('Importance Score')
    ax1.set_title('Frequency Band Importance by Emotion Class')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], rotation=15)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Stacked/normalized comparison
    ax2 = axes[1]
    band_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    for idx, class_name in enumerate(class_names):
        if class_name not in results or results[class_name] is None:
            continue
        mean_feat = results[class_name]['mean_feat']
        # Normalize to show relative importance
        normalized = mean_feat / mean_feat.sum() * 100
        ax2.barh(class_name, normalized, left=0, color=BAND_COLORS, height=0.6)

    # Create legend for bands
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BAND_COLORS[i], label=band_labels[i])
                       for i in range(len(band_labels))]
    ax2.legend(handles=legend_elements, loc='lower right', title='Bands')
    ax2.set_xlabel('Relative Importance (%)')
    ax2.set_title('Normalized Frequency Band Contribution')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance plot to {save_path}")


def print_feature_comparison(results):
    """Print feature importance comparison across classes."""
    class_names = ['Negative', 'Neutral', 'Positive']

    print("\n" + "="*60)
    print("FREQUENCY BAND IMPORTANCE COMPARISON")
    print("="*60)

    # Create comparison table
    print("\n{:<12} {:>12} {:>12} {:>12}".format(
        "Band", "Negative", "Neutral", "Positive"))
    print("-" * 50)

    for band_idx, band_name in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']):
        values = []
        for class_name in class_names:
            if class_name in results and results[class_name] is not None:
                val = results[class_name]['mean_feat'][band_idx]
                values.append(f"{val:.4f}")
            else:
                values.append("N/A")
        print("{:<12} {:>12} {:>12} {:>12}".format(band_name, *values))

    # Find most important band per class
    print("\nMost important band per class:")
    for class_name in class_names:
        if class_name in results and results[class_name] is not None:
            mean_feat = results[class_name]['mean_feat']
            top_band_idx = np.argmax(mean_feat)
            band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
            print(f"  {class_name}: {band_names[top_band_idx]} ({mean_feat[top_band_idx]:.4f})")


def print_cross_class_comparison(results):
    """Print edges that are important across multiple classes or unique to one class."""
    class_names = ['Negative', 'Neutral', 'Positive']

    # Collect top 20 edges per class
    class_top_edges = {}
    for class_name in class_names:
        if class_name in results and results[class_name] is not None:
            edges = results[class_name]['top_edges']
            class_top_edges[class_name] = set((e[0], e[1]) for e in edges)

    if len(class_top_edges) < 2:
        return

    print("\n" + "="*60)
    print("CROSS-CLASS EDGE COMPARISON")
    print("="*60)

    # Find common edges (in all classes)
    all_edges = set.intersection(*class_top_edges.values()) if class_top_edges else set()
    if all_edges:
        print(f"\nEdges important for ALL classes ({len(all_edges)}):")
        for src, dst in sorted(all_edges):
            print(f"  Electrode {src} <-> Electrode {dst}")

    # Find class-specific edges
    for class_name in class_names:
        if class_name not in class_top_edges:
            continue
        other_edges = set.union(*[v for k, v in class_top_edges.items() if k != class_name]) if len(class_top_edges) > 1 else set()
        unique = class_top_edges[class_name] - other_edges
        if unique:
            print(f"\nEdges unique to {class_name} ({len(unique)}):")
            for src, dst in sorted(unique):
                print(f"  Electrode {src} <-> Electrode {dst}")


def main():
    args = Args()

    # Setup device
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create log directory
    os.makedirs(args.logdir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = load_model(args, device)

    # Load dataset
    print("Loading dataset...")
    dataset = load_data()

    # Get samples by class
    class_samples = get_samples_by_class(
        dataset, model, device,
        samples_per_class=args.samples_per_class,
        correct_only=args.correct_only
    )

    class_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    # Filter classes if specified
    classes_to_analyze = args.class_filter if args.class_filter else [0, 1, 2]

    # Run explanations for each class
    results = {}
    for class_idx in classes_to_analyze:
        class_name = class_names[class_idx]
        samples = class_samples.get(class_idx, [])

        if not samples:
            print(f"\nNo samples found for class {class_name}")
            continue

        masked_adjs, feat_masks, samples = run_explanations_for_class(
            model, samples, args, device, class_name
        )

        results[class_name] = analyze_class_explanations(
            masked_adjs, feat_masks, samples, class_name, top_k=args.top_k
        )

    # Cross-class comparison
    print_cross_class_comparison(results)
    print_feature_comparison(results)

    # Save aggregated plots
    plot_aggregated_results(results, os.path.join(args.logdir, 'aggregated_importance.png'))
    plot_top_edges_comparison(results, os.path.join(args.logdir, 'top_edges_comparison.png'),
                              top_k=args.top_k)
    plot_connectivity_circles(results, args.logdir, n_lines=args.n_lines)
    plot_feature_importance(results, os.path.join(args.logdir, 'feature_importance.png'))

    # Save results to numpy
    for class_name, data in results.items():
        if data is not None:
            np.save(os.path.join(args.logdir, f'{class_name.lower()}_mean_adj.npy'), data['mean_adj'])
            np.save(os.path.join(args.logdir, f'{class_name.lower()}_std_adj.npy'), data['std_adj'])
            np.save(os.path.join(args.logdir, f'{class_name.lower()}_mean_feat.npy'), data['mean_feat'])
            np.save(os.path.join(args.logdir, f'{class_name.lower()}_std_feat.npy'), data['std_feat'])

    print("\n" + "="*60)
    print("EXPLANATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.logdir}")


if __name__ == '__main__':
    main()

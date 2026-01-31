"""
Main script for DGCNN EEG analysis with GNNExplainer.

This script:
1. Loads the SEED dataset and trained DGCNN model
2. Converts the model to PyG-compatible format for explainability
3. Runs standard GNNExplainer and contrastive explanations
4. Generates visualizations and validation metrics
"""

import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader

from torch_geometric.explain import Explainer, GNNExplainer
from torcheeg.datasets import SEEDDataset
from torcheeg import transforms
from torcheeg.models import DGCNN

from contrastive_explainer import explain_class_contrast
from plotting import visualize_node_importance_subplots

from convertDGCNN import (
    prepare_for_explainer_pyg,
    get_all_class_prototypes,
    compute_all_validation_metrics,
    print_validation_metrics,
)

from plotting import (
    visualize_learned_adjacency,
    visualize_edge_importance_circular_subplots,
    plot_topomap_node_importance_subplots,
    plot_sparsity_curve,
    plot_validation_summary,
)


def main():
    # Configuration
    THRESHOLD = 0.0
    N_SAMPLES_PER_CLASS = 1000
    N_LINES = 10
    BATCH_SIZE = 2500
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

    visualize_learned_adjacency(
        model, threshold=THRESHOLD, save_path=f"{OUTPUT_DIR}/learned_adjacency.png"
    )

    # Prepare PyG-compatible model
    explainer_model, edge_index, edge_attr = prepare_for_explainer_pyg(
        model, threshold=THRESHOLD
    )

    # Get normalized edge weights
    edge_weight = explainer_model._get_normalized_edge_weights(edge_index)

    # Setup GNNExplainer
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

    class_names = ["Negative", "Neutral", "Positive"]

    all_standard_metrics = {}

    def _aggregate_node_mask_for_topomap(node_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert node_mask to a single per-node importance vector [num_nodes].
        If mask is [num_nodes, num_features], aggregate over features.
        """
        if not torch.is_tensor(node_mask):
            node_mask = torch.as_tensor(node_mask)

        m = node_mask.detach()
        # common shapes: [62], [62, 5], sometimes with singleton dims
        m = m.squeeze()

        if m.dim() == 2:
            # aggregate feature/channel importance into one scalar per node
            m = m.abs().mean(dim=-1)
        elif m.dim() != 1:
            # last-resort: flatten then take mean per node if possible isn't safe; fail loudly
            raise ValueError(
                f"Unexpected node_mask shape for topomap: {tuple(m.shape)}"
            )

        return m.cpu()

    # =========================================================================
    # STANDARD GNN EXPLAINER - CLASS PROTOTYPES
    # =========================================================================


    N_PROTOTYPE_SAMPLES = N_SAMPLES_PER_CLASS

    standard_prototypes = get_all_class_prototypes(
        model=explainer_model,
        explainer=explainer,
        data_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_classes=3,
        num_samples_per_class=N_PROTOTYPE_SAMPLES,
        use_contrastive=False,
    )

    # Visualize standard prototype explanations
    print("\nVisualizing standard class prototype explanations...")
    standard_node_masks = {
        idx: p["prototype_node_mask"] for idx, p in standard_prototypes.items()
    }
    standard_edge_masks = {
        idx: p["prototype_edge_mask"] for idx, p in standard_prototypes.items()
    }

    standard_node_masks_agg = {
        idx: _aggregate_node_mask_for_topomap(mask)
        for idx, mask in standard_node_masks.items()
    }

    visualize_node_importance_subplots(
        node_masks=standard_node_masks_agg,
        class_names=class_names,
        title="Standard Node Importance (Aggregated Features)",
        save_path=f"{OUTPUT_DIR}/standard_node_all_agg.png",
        normalize_per_plot=True,
    )

    plot_topomap_node_importance_subplots(
        node_masks=standard_node_masks_agg,
        class_names=class_names,
        title="Standard GNNExplainer - Node Importance (Aggregated Features, Topomap)",
        save_path=f"{OUTPUT_DIR}/standard_prototype_topomap_node_agg.png",
    )

    visualize_edge_importance_circular_subplots(
        edge_index=edge_index,
        edge_masks=standard_edge_masks,
        class_names=class_names,
        num_electrodes=62,
        n_lines=N_LINES,
        title="Standard Edge Importance (Median)",
        save_path=f"{OUTPUT_DIR}/standard_prototype_edge.png",
        normalize_per_plot=True,
    )

    # =========================================================================
    # VALIDATION METRICS - STANDARD GNN EXPLAINER (PER-CLASS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION METRICS - STANDARD GNN EXPLAINER")
    print("=" * 70)

    standard_samples_per_class = 50  
    standard_class_samples = {0: [], 1: [], 2: []}

    for x, y in test_loader:
        for i in range(len(y)):
            c = int(y[i].item())
            if len(standard_class_samples[c]) < standard_samples_per_class:
                standard_class_samples[c].append(x[i])
        if all(
            len(v) >= standard_samples_per_class
            for v in standard_class_samples.values()
        ):
            break

    for class_idx, class_name in enumerate(class_names):
        samples_for_validation = []
        edge_masks_for_validation = []

        for sample in standard_class_samples[class_idx]:
            sample = sample.squeeze()

            # Standard (non-contrastive) explanation for this specific class target
            explanation = explainer(
                x=sample,
                edge_index=edge_index,
                edge_weight=edge_weight,
                target=class_idx,
            )

            samples_for_validation.append((sample, class_idx))
            edge_masks_for_validation.append(explanation.edge_mask.detach().cpu())

        metrics = compute_all_validation_metrics(
            explainer_model,
            samples_for_validation,
            edge_masks_for_validation,
            edge_index,
            device=str(DEVICE),
        )
        all_standard_metrics[class_name] = metrics
        print_validation_metrics(metrics, f"Standard - {class_name}")

    if all_standard_metrics:
        sparsity_results = {
            name: m["sparsity_curve"] for name, m in all_standard_metrics.items()
        }
        plot_sparsity_curve(
            sparsity_results,
            list(all_standard_metrics.keys()),
            save_path=f"{OUTPUT_DIR}/sparsity_curve_standard.png",
        )

    # =========================================================================
    # CONTRASTIVE GNN EXPLAINER (PAIRWISE)
    # =========================================================================

    contrasts = [
        (0, 2, "Negative vs Positive"),
        (2, 0, "Positive vs Negative"),
        (0, 1, "Negative vs Neutral"),
        (1, 0, "Neutral vs Negative"),
        (1, 2, "Neutral vs Positive"),
        (2, 1, "Positive vs Neutral"),
    ]

    # Collect multiple samples per class for contrastive validation
    contrastive_samples_per_class = 50
    class_samples_list = {0: [], 1: [], 2: []}
    for x, y in test_loader:
        for i in range(len(y)):
            label = y[i].item()
            if len(class_samples_list[label]) < contrastive_samples_per_class:
                class_samples_list[label].append(x[i])
        if all(
            len(v) >= contrastive_samples_per_class for v in class_samples_list.values()
        ):
            break

    contrastive_node_masks = {}
    contrastive_edge_masks = {}
    contrast_labels = []
    contrast_keys_in_order = []
    contrastive_validation_data = {}

    for target_class, contrast_class, description in contrasts:
        print(f"\n{'='*60}")
        print(f"Contrastive Explanation: {description}")
        print(f"  Target class: {class_names[target_class]}")
        print(f"  Contrast class: {class_names[contrast_class]}")
        print(f"{'='*60}")

        key = f"{target_class}_vs_{contrast_class}"
        contrast_keys_in_order.append(key)
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
            "samples": samples_for_validation,
            "edge_masks": edge_masks_for_validation,
        }

    # Visualize all contrastive explanations as subplots
    print("\nGenerating contrastive subplot visualizations...")

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

    visualize_node_importance_subplots(
        node_masks=contrastive_node_masks,
        class_names=contrast_labels,
        title="Contrastive Node Importance",
        save_path=f"{OUTPUT_DIR}/contrastive_node_all.png",
        normalize_per_plot=True,
    )

    contrastive_node_masks_for_topomap = {
        i: _aggregate_node_mask_for_topomap(contrastive_node_masks[k])
        for i, k in enumerate(contrast_keys_in_order)
        if k in contrastive_node_masks
    }

    plot_topomap_node_importance_subplots(
        node_masks=contrastive_node_masks_for_topomap,
        class_names=contrast_labels,  # 6 labels -> 2x3 subplots
        title="Contrastive GNNExplainer - Node Importance (Aggregated Features, Topomap)",
        save_path=f"{OUTPUT_DIR}/contrastive_topomap_node_all_agg.png",
    )

    # =========================================================================
    # VALIDATION METRICS - CONTRASTIVE GNN EXPLAINER (PAIRWISE)
    # =========================================================================

    print("\n" + "=" * 70)
    print("VALIDATION METRICS - CONTRASTIVE GNN EXPLAINER (PAIRWISE)")
    print("=" * 70)

    all_contrastive_metrics = {}
    for label, data in contrastive_validation_data.items():
        metrics = compute_all_validation_metrics(
            explainer_model,
            data["samples"],  
            data["edge_masks"], 
            edge_index,
            device=str(DEVICE),
        )
        all_contrastive_metrics[label] = metrics
        print_validation_metrics(metrics, f"Contrastive - {label}")

    if all_contrastive_metrics:
        sparsity_results = {
            name: m["sparsity_curve"] for name, m in all_contrastive_metrics.items()
        }
        plot_sparsity_curve(
            sparsity_results,
            list(all_contrastive_metrics.keys()),
            save_path=f"{OUTPUT_DIR}/sparsity_curve_contrastive.png",
        )

    # =========================================================================
    # COMPARISON: STANDARD vs CONTRASTIVE
    # =========================================================================

    print("\n" + "=" * 70)
    print("COMPARISON: STANDARD vs CONTRASTIVE")
    print("=" * 70)

    combined_metrics = {}
    for name, metrics in all_standard_metrics.items():
        combined_metrics[f"Std-{name}"] = metrics
    for name, metrics in all_contrastive_metrics.items():
        combined_metrics[f"Con-{name}"] = metrics

    if combined_metrics:
        plot_validation_summary(
            combined_metrics,
            save_path=f"{OUTPUT_DIR}/validation_summary_comparison.png",
        )


if __name__ == "__main__":
    main()

"""test_converted_dgcnn.py

Script to verify that the converted PyG-compatible DGCNN model produces
the same outputs as the original DGCNN model.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torcheeg.datasets import SEEDDataset
from torcheeg import transforms
from torcheeg.models import DGCNN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from convertDGCNN import prepare_for_explainer_pyg, DGCNNForExplainerPyG


# Config
BATCH_SIZE = 256
MODEL_PATH = "ckpts/dgcnn_seed_model.pth"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def load_data():
    """Load the SEED dataset with same split as training."""
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
            transforms.Lambda(lambda x: x + 1)  # SEED labels: -1,0,1 -> 0,1,2
        ])
    )

    # Use same split as training (seed=0)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(0)
    )

    return dataset, train_dataset, test_dataset


def load_models(device):
    """Load both original and converted DGCNN models."""
    # Original model
    original_model = DGCNN(
        in_channels=5,
        num_electrodes=62,
        hid_channels=32,
        num_layers=2,
        num_classes=3
    )
    original_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    original_model = original_model.to(device)
    original_model.eval()

    # Converted PyG-compatible model
    converted_model, edge_index, edge_attr = prepare_for_explainer_pyg(original_model, threshold=0.0)
    converted_model = converted_model.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    return original_model, converted_model, edge_index, edge_attr


def evaluate_original(model, dataloader, device):
    """Evaluate original model and return predictions and labels."""
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs), np.array(all_logits)


def evaluate_converted(model, dataloader, edge_index, edge_attr, device):
    """Evaluate converted PyG model and return predictions and labels."""
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []

    # Get normalized edge weights
    edge_weight = model._get_normalized_edge_weights(edge_index)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Process each sample individually (converted model expects single samples)
            batch_outputs = []
            for i in range(x.shape[0]):
                sample = x[i].squeeze()  # [62, 5]
                output = model(sample, edge_index, edge_weight)
                batch_outputs.append(output)

            outputs = torch.cat(batch_outputs, dim=0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs), np.array(all_logits)


def compare_outputs(original_logits, converted_logits, tolerance=1e-4):
    """Compare outputs between original and converted models."""
    diff = np.abs(original_logits - converted_logits)
    max_diff = diff.max()
    mean_diff = diff.mean()

    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'within_tolerance': max_diff < tolerance,
        'num_mismatches': (diff > tolerance).sum()
    }


def main():
    print(f"Using device: {DEVICE}")

    # Load data
    print("Loading dataset...")
    dataset, train_dataset, test_dataset = load_data()

    print(f"\nDataset sizes:")
    print(f"  Total: {len(dataset)}")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Test:  {len(test_dataset)}")

    # Load models
    print("\nLoading models...")
    original_model, converted_model, edge_index, edge_attr = load_models(DEVICE)

    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge attr shape: {edge_attr.shape}")
    num_self_loops = (edge_index[0] == edge_index[1]).sum().item()
    print(f"Number of edges (including {num_self_loops} self-loops): {edge_index.shape[1]}")

    # Create dataloaders
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluate original model
    print("\n" + "="*60)
    print("ORIGINAL MODEL EVALUATION")
    print("="*60)
    orig_preds, orig_labels, orig_probs, orig_logits = evaluate_original(
        original_model, test_loader, DEVICE
    )
    orig_acc = accuracy_score(orig_labels, orig_preds)
    print(f"\nAccuracy: {orig_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(orig_labels, orig_preds,
                                target_names=['Negative (0)', 'Neutral (1)', 'Positive (2)']))

    # Evaluate converted model
    print("\n" + "="*60)
    print("CONVERTED (PyG) MODEL EVALUATION")
    print("="*60)
    conv_preds, conv_labels, conv_probs, conv_logits = evaluate_converted(
        converted_model, test_loader, edge_index, edge_attr, DEVICE
    )
    conv_acc = accuracy_score(conv_labels, conv_preds)
    print(f"\nAccuracy: {conv_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(conv_labels, conv_preds,
                                target_names=['Negative (0)', 'Neutral (1)', 'Positive (2)']))

    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    # Prediction agreement
    agreement = (orig_preds == conv_preds).sum() / len(orig_preds) * 100
    print(f"\nPrediction agreement: {agreement:.2f}%")
    print(f"Number of differing predictions: {(orig_preds != conv_preds).sum()}")

    # Output comparison
    comparison = compare_outputs(orig_logits, conv_logits)
    print(f"\nLogit comparison:")
    print(f"  Max difference: {comparison['max_diff']:.6f}")
    print(f"  Mean difference: {comparison['mean_diff']:.6f}")
    print(f"  Within tolerance (1e-4): {comparison['within_tolerance']}")

    if comparison['within_tolerance']:
        print("\n✓ Converted model produces identical outputs to original!")
    else:
        print(f"\n✗ Found {comparison['num_mismatches']} logit values outside tolerance")
        print("  Note: Small differences may be due to numerical precision in message passing")

    # Confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_orig = confusion_matrix(orig_labels, orig_preds)
    cm_conv = confusion_matrix(conv_labels, conv_preds)

    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'Original DGCNN (Acc: {orig_acc*100:.2f}%)')

    sns.heatmap(cm_conv, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title(f'Converted PyG DGCNN (Acc: {conv_acc*100:.2f}%)')

    plt.tight_layout()
    plt.savefig('plots/confusion_matrix_comparison.png', dpi=150)
    print("\nConfusion matrix comparison saved to plots/confusion_matrix_comparison.png")

    # Per-class comparison
    print("\n" + "="*60)
    print("PER-CLASS COMPARISON (Test Set)")
    print("="*60)
    for i, label_name in enumerate(['Negative', 'Neutral', 'Positive']):
        mask = orig_labels == i
        orig_class_acc = (orig_preds[mask] == i).sum() / mask.sum() * 100
        conv_class_acc = (conv_preds[mask] == i).sum() / mask.sum() * 100
        print(f"\n{label_name} (class {i}):")
        print(f"  Samples: {mask.sum()}")
        print(f"  Original accuracy: {orig_class_acc:.2f}%")
        print(f"  Converted accuracy: {conv_class_acc:.2f}%")
        print(f"  Difference: {abs(orig_class_acc - conv_class_acc):.2f}%")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original model accuracy:  {orig_acc*100:.2f}%")
    print(f"Converted model accuracy: {conv_acc*100:.2f}%")
    print(f"Prediction agreement:     {agreement:.2f}%")
    print(f"Max logit difference:     {comparison['max_diff']:.6f}")

    if agreement == 100.0:
        print("\n✓ Models are functionally equivalent!")
    elif agreement > 99.0:
        print("\n~ Models are nearly equivalent (>99% agreement)")
    else:
        print(f"\n! Models have significant differences ({100-agreement:.2f}% disagreement)")


if __name__ == '__main__':
    main()

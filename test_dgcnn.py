"""test_dgcnn.py

Script to evaluate the trained DGCNN model on the test set.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torcheeg.datasets import SEEDDataset
from torcheeg import transforms
from torcheeg.models import DGCNN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# Config
BATCH_SIZE = 256
MODEL_PATH = "ckpts/dgcnn_seed_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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


def load_model(device):
    """Load the trained DGCNN model."""
    model = DGCNN(
        in_channels=5,
        num_electrodes=62,
        hid_channels=32,
        num_layers=2,
        num_classes=3
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate(model, dataloader, device):
    """Evaluate model and return predictions and labels."""
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    print(f"Using device: {DEVICE}")

    # Load data
    print("Loading dataset...")
    dataset, train_dataset, test_dataset = load_data()

    print(f"\nDataset sizes:")
    print(f"  Total: {len(dataset)}")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Test:  {len(test_dataset)}")

    # Load model
    print("\nLoading model...")
    model = load_model(DEVICE)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluate on train set
    print("\n" + "="*60)
    print("TRAIN SET EVALUATION")
    print("="*60)
    train_preds, train_labels, train_probs = evaluate(model, train_loader, DEVICE)
    train_acc = accuracy_score(train_labels, train_preds)
    print(f"\nAccuracy: {train_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(train_labels, train_preds,
                                target_names=['Negative (0)', 'Neutral (1)', 'Positive (2)']))

    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    test_preds, test_labels, test_probs = evaluate(model, test_loader, DEVICE)
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"\nAccuracy: {test_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=['Negative (0)', 'Neutral (1)', 'Positive (2)']))

    # Confusion matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)

    # Per-class analysis
    print("\n" + "="*60)
    print("PER-CLASS ANALYSIS (Test Set)")
    print("="*60)
    for i, label_name in enumerate(['Negative', 'Neutral', 'Positive']):
        mask = test_labels == i
        class_acc = (test_preds[mask] == i).sum() / mask.sum() * 100
        print(f"\n{label_name} (class {i}):")
        print(f"  Samples: {mask.sum()}")
        print(f"  Accuracy: {class_acc:.2f}%")
        print(f"  Predicted distribution: {np.bincount(test_preds[mask], minlength=3)}")

    # Confidence analysis
    print("\n" + "="*60)
    print("CONFIDENCE ANALYSIS (Test Set)")
    print("="*60)
    max_probs = test_probs.max(axis=1)
    print(f"\nMean confidence: {max_probs.mean()*100:.2f}%")
    print(f"Confidence on correct: {max_probs[test_preds == test_labels].mean()*100:.2f}%")
    print(f"Confidence on incorrect: {max_probs[test_preds != test_labels].mean()*100:.2f}%")

    # Class distribution check
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)
    print(f"\nTrain labels distribution: {np.bincount(train_labels)}")
    print(f"Test labels distribution:  {np.bincount(test_labels)}")
    print(f"Test predictions distribution: {np.bincount(test_preds)}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Test Acc: {test_acc*100:.2f}%)')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    print("\nConfusion matrix saved to confusion_matrix.png")



if __name__ == '__main__':
    main()

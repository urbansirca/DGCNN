"""explain_dgcnn.py

Script to explain DGCNN model predictions using the GNN Explainer.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torcheeg.datasets import SEEDDataset
from torcheeg import transforms
from torcheeg.models import DGCNN
from tensorboardX import SummaryWriter

from explainer.explain import Explainer


class Args:
    """Configuration class with hardcoded parameters."""
    # Model parameters
    model_path = "ckpts/dgcnn_seed_model.pth"
    in_channels = 5
    num_electrodes = 62
    hid_channels = 32
    num_layers = 1
    num_classes = 3

    # Explainer parameters
    num_epochs = 100
    num_gc_layers = 1
    mask_act = 'sigmoid'
    mask_bias = False

    # Naming parameters (used by io_utils.gen_prefix)
    bmname = None  # If None, uses dataset name
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


def main():
    args = Args()

    # Setup device
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create log directory
    os.makedirs(args.logdir, exist_ok=True)

    # Setup tensorboard writer
    writer = SummaryWriter(args.logdir)

    # Load model
    print("Loading model...")
    model = load_model(args, device)

    # Load dataset
    print("Loading dataset...")
    dataset = load_data()

    # Get samples to explain
    num_samples = min(args.num_samples, len(dataset))

    # Prepare data for explainer (graph-level explanation)
    print(f"Preparing {num_samples} samples for explanation...")

    features_list = []
    labels_list = []
    preds_list = []

    for i in range(num_samples):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            pred = model(x)
            pred = torch.softmax(pred, dim=1)

        features_list.append(x.cpu().numpy())
        labels_list.append(y)
        preds_list.append(pred.cpu().numpy())

    # Stack features and labels
    feat = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)
    pred = np.array(preds_list)

    # Get adjacency from the model (learned adjacency matrix)
    adj = model.A.detach().cpu().numpy()
    adj = np.tile(adj, (num_samples, 1, 1))

    print(f"Feature shape: {feat.shape}")
    print(f"Adjacency shape: {adj.shape}")
    print(f"Labels: {labels}")
    print(f"Predictions shape: {pred.shape}")

    # Create explainer
    print("Creating explainer...")
    explainer = Explainer(
        model=model,
        adj=adj,
        feat=feat,
        label=labels,
        pred=pred,
        train_idx=np.arange(num_samples),
        args=args,
        writer=writer,
        print_training=True,
        graph_mode=True,
        graph_idx=args.graph_idx
    )

    # Run explanation for each sample
    print("Running explanations...")
    graph_indices = list(range(num_samples))
    masked_adjs = explainer.explain_graphs(graph_indices)

    print("\nExplanation complete!")
    print(f"Results saved to: {args.logdir}")

    # Print summary of important edges
    for i, masked_adj in enumerate(masked_adjs):
        print(f"\nSample {i} (Label: {labels[i]}, Pred: {np.argmax(pred[i])}):")
        important_edges = np.where(masked_adj > np.percentile(masked_adj, 90))
        print(f"  Top 10% important edges: {len(important_edges[0])} edges")

        # Get top 5 most important edges
        flat_idx = np.argsort(masked_adj.flatten())[-5:]
        top_edges = np.unravel_index(flat_idx, masked_adj.shape)
        print("  Top 5 edges (electrode pairs):")
        for j in range(5):
            src, dst = top_edges[0][j], top_edges[1][j]
            weight = masked_adj[src, dst]
            print(f"    Electrode {src} <-> Electrode {dst}: {weight:.4f}")

    writer.close()


if __name__ == '__main__':
    main()

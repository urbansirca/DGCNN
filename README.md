# GNNExplainer for DGCNN on EEG Emotion Recognition

This project applies GNNExplainer to the Dynamical Graph Convolutional Neural Network (DGCNN) for EEG-based emotion recognition on the SEED dataset.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download SEED dataset

1. Request access from: https://bcmi.sjtu.edu.cn/home/seed/
2. Download the "Preprocessed_EEG" folder
3. Extract to `./Preprocessed_EEG/`

The folder structure should be:
```
Preprocessed_EEG/
├── 1/
│   ├── 1_20131027.mat
│   ├── 1_20131030.mat
│   └── 1_20131107.mat
├── 2/
│   └── ...
└── ...
```

## Usage

### Train DGCNN

**Quick test (one fold, 5 epochs):**
```bash
python train_dgcnn_seed.py --quick
```

**Full leave-one-subject-out cross-validation:**
```bash
python train_dgcnn_seed.py --epochs 50
```

This will:
- Train 15 models (one per fold)
- Save best model for each subject in `./checkpoints/`
- Report mean accuracy across subjects

Expected results: ~75-80% accuracy (subject-independent)

### Generate Explanations

**Standard GNNExplainer:**
```bash
python gnnexplainer_dgcnn.py \
    --checkpoint checkpoints/dgcnn_subject_1_best.pt \
    --num_samples 50 \
    --output_dir explanations/standard
```

**Contrastive explanations:**
```bash
python gnnexplainer_dgcnn.py \
    --checkpoint checkpoints/dgcnn_subject_1_best.pt \
    --num_samples 50 \
    --contrastive \
    --output_dir explanations/contrastive
```

### Output Files

After running the explainer:
- `edge_masks.npy`: Edge importance for each sample [N, 62, 62]
- `node_masks.npy`: Node feature importance for each sample [N, 62, 5]
- `mean_edge_mask.npy`: Aggregated edge importance [62, 62]
- `mean_node_mask.npy`: Aggregated node importance [62, 5]
- `edge_importance.png`: Visualisation of edge importance
- `node_importance.png`: Visualisation per electrode and frequency band

## Project Structure

```
.
├── train_dgcnn_seed.py      # Training script
├── gnnexplainer_dgcnn.py    # Custom GNNExplainer implementation
├── requirements.txt
├── README.md
├── Preprocessed_EEG/        # SEED dataset (download separately)
├── seed_de_cache/           # Cached preprocessed data (auto-generated)
├── checkpoints/             # Saved models
└── explanations/            # Generated explanations
```

## Key Implementation Details

### Why custom GNNExplainer?

PyTorch Geometric's GNNExplainer expects:
- Sparse `edge_index` format: `[2, num_edges]`
- Message-passing layers (GCNConv, GATConv, etc.)

DGCNN uses:
- Dense adjacency matrix: `[62, 62]`
- Chebyshev spectral convolution

Our implementation directly masks the dense adjacency matrix during forward passes.

### Contrastive Objective

Standard: "Why class X?"
```
L = -log P(Y=x | masked_graph) + sparsity + entropy
```

Contrastive: "Why class X instead of Y?"
```
L = -log P(Y=x | masked_graph) + log P(Y=y | masked_graph) + sparsity + entropy
```

This finds edges that discriminate between emotions, not just predict one.

## References

- Song et al. (2020). "EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks." IEEE TAFFC.
- Ying et al. (2019). "GNNExplainer: Generating Explanations for Graph Neural Networks." NeurIPS.
- Zhdanov et al. (2022). "Investigating Brain Connectivity with Graph Neural Networks and GNNExplainer." ICPR.

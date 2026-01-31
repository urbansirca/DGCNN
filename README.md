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

This project has **two main entry points**:

### 1. Main Pipeline (EEG + Contrastive Extensions)

This is the primary workflow for training DGCNN on EEG data and generating contrastive explanations.

#### Step 1: Train DGCNN Model
```bash
python train_dgcnn.py
```

This will:
- Train DGCNN on the SEED EEG dataset
- Save the best model to `ckpts/dgcnn_seed_model.pth`
- Expected accuracy: ~90% on test set

#### Step 2: Generate Explanations
```bash
python main.py
```

This will:
- Load the trained DGCNN model from `ckpts/dgcnn_seed_model.pth`
- Convert the model to PyG-compatible format
- Generate **standard GNNExplainer** explanations (per-class prototypes)
- Generate **contrastive GNNExplainer** explanations (pairwise class comparisons)
- Compute validation metrics (fidelity, sparsity, stability)
- Save visualizations to `plots/` directory

**Output files in `plots/`:**
- `learned_adjacency.png`: Learned adjacency matrix from DGCNN
- `standard_node_all_agg.png`: Standard node importance (aggregated features)
- `standard_prototype_topomap_node_agg.png`: Topographic maps of node importance
- `standard_prototype_edge.png`: Standard edge importance visualizations
- `contrastive_edge_all.png`: Contrastive edge importance (6 pairwise comparisons)
- `contrastive_node_all.png`: Contrastive node importance
- `contrastive_topomap_node_all_agg.png`: Topographic maps for contrastive explanations
- `sparsity_curve_standard.png`: Sparsity vs. fidelity curves for standard GNNExplainer
- `sparsity_curve_contrastive.png`: Sparsity vs. fidelity curves for contrastive GNNExplainer
- `validation_summary_comparison.png`: Comparison of standard vs. contrastive metrics

### 2. GNNExplainer Reproduction (Synthetic Datasets)

For reproducing the original GNNExplainer paper results on synthetic datasets, see the `reproduction/` folder. This contains a minimal implementation on synthetic graph datasets (syn1, syn3).

```bash
cd reproduction
python run_gnn_explainer.py --dataset syn1 --node-idx 300
```

See [reproduction/README.md](reproduction/README.md) for more details.

## Project Structure

```
.
├── train_dgcnn.py           # [Step 1] Train DGCNN on SEED dataset
├── main.py                  # [Step 2] Generate explanations with contrastive loss
├── convertDGCNN.py          # Convert DGCNN to PyG-compatible format
├── contrastive_explainer.py # Contrastive GNNExplainer implementation
├── plotting.py              # Visualization utilities
├── requirements.txt
├── README.md
├── Preprocessed_EEG/        # SEED dataset (download separately)
├── ckpts/                   # Saved models
├── plots/                   # Generated visualizations
└── reproduction/            # GNNExplainer reproduction on synthetic datasets
    ├── run_gnn_explainer.py
    ├── gnnexplainer.py
    └── models.py
```


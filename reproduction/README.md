# GNNExplainer Reproduction on Synthetic Datasets

This folder contains a minimal reproduction of the original GNNExplainer paper ([Ying et al., NeurIPS 2019](https://arxiv.org/abs/1903.03894)) on synthetic graph datasets.

## Overview

This implementation demonstrates GNNExplainer on simple synthetic datasets where ground-truth explanations are known, making it easy to validate that the explainer is working correctly.

## Files

- **run_gnn_explainer.py**: Main script to run GNNExplainer on synthetic datasets
- **gnnexplainer.py**: Core GNNExplainer implementation
- **models.py**: GNN model architectures (GCN encoder for node/graph classification)

## Prerequisites

You'll need pre-trained model checkpoints for the synthetic datasets. These should be located in `../ckpts/` directory.

Expected checkpoint structure:
```
../ckpts/
├── syn1_base_h20_o20/
│   └── best.pth.tar
└── syn3_base_h20_o20/
    └── best.pth.tar
```
you will need to train models first (training script not included in this reproduction).

## Usage


Explain a specific node in the graph:

```bash
python run_gnn_explainer.py --dataset syn1 --node-idx 300
```

**Parameters:**
- `--dataset`: Dataset name (`syn1`, `syn3`)
- `--node-idx`: Index of the node to explain (default: 300)
- `--num-epochs`: Number of optimization epochs (default: 100)
- `--lr`: Learning rate for mask optimization (default: 0.1)
- `--threshold`: Visualization threshold for edges (default: 0.1)
- `--top-k`: Keep only top-k edges in explanation (optional)
- `--size-coeff`: Sparsity regularization coefficient (default: 0.005, higher = sparser)
- `--contrast-coeff`: Contrastive loss coefficient (default: 0.0)


## Output

After running, the script will generate:

1. **Console output:**
   - Node/graph label and prediction
   - Subgraph size (for node classification)
   - Number of non-zero edges in the explanation
   - Top 5 important features

2. **Visualization:** `explanation.png`
   - Graph visualization showing important edges
   - Target node highlighted in red
   - Edge thickness/color indicates importance

## References

- Ying et al. (2019). "GNNExplainer: Generating Explanations for Graph Neural Networks." NeurIPS.

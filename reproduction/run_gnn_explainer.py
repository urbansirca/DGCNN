import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

import models
from gnnexplainer import GNNExplainer, extract_neighborhood


def load_checkpoint(ckpt_dir, dataset, hidden_dim=20, output_dim=20, method="base"):
    """Load a pre-trained model checkpoint from ../ckpts relative to this script."""
    import os

    # Get the directory one level up from this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    ckpt_base_dir = os.path.join(parent_dir, ckpt_dir)

    # Construct checkpoint name
    name = f"{dataset}_{method}_h{hidden_dim}_o{output_dim}"

    # Try different possible paths
    filename =  os.path.join(ckpt_base_dir, f"{name}.pth.tar")
    

    print(f"Loading checkpoint: {filename}")
    ckpt = torch.load(filename, map_location="cpu", weights_only=False)
    return ckpt


def build_model(
    input_dim,
    num_classes,
    hidden_dim,
    output_dim,
    num_layers,
    use_gpu=True,
):
    """Build the GNN model for node classification."""

    # Create a minimal args object for the model
    class Args:
        def __init__(self):
            self.gpu = use_gpu
            self.bias = True
            self.method = "base"

    args = Args()

    model = models.GcnEncoderNode(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=output_dim,
        label_dim=num_classes,
        num_layers=num_layers,
        bn=False,
        args=args,
    )

    if use_gpu and torch.cuda.is_available():
        model = model.cuda()

    return model


def visualize_explanation(
    masked_adj, node_idx=None, threshold=0.1, title="Explanation"
):
    # Threshold the adjacency
    adj_thresh = masked_adj.copy()
    adj_thresh[adj_thresh < threshold] = 0

    # Create graph
    G = nx.from_numpy_array(adj_thresh)

    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    if G.number_of_nodes() == 0:
        print("No edges above threshold. Try lowering the threshold.")
        return

    # Get edge weights for coloring
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    node_colors = ["red" if n == node_idx else "lightblue" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)

    # Draw edges with weights as colors
    if weights:
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=weights,
            edge_cmap=plt.cm.Blues,
            width=2,
            edge_vmin=0,
            edge_vmax=max(weights),
        )

    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("explanation.png", dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Run GNNExplainer on synthetic node classification datasets (syn1, syn3)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="syn1",
        help="Dataset name (syn1, syn3)",
    )
    parser.add_argument(
        "--ckptdir", type=str, default="ckpts", help="Checkpoint directory"
    )
    parser.add_argument(
        "--node-idx",
        type=int,
        default=300,
        help="Node index to explain",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=20, help="Hidden dimension of the model"
    )
    parser.add_argument(
        "--output-dim", type=int, default=20, help="Output dimension of the model"
    )
    parser.add_argument(
        "--num-gc-layers", type=int, default=3, help="Number of graph conv layers"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of optimization epochs for explainer",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for explainer (original default: 0.1)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Threshold for visualization"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep only top-k edges in explanation (for sparser results)",
    )
    parser.add_argument(
        "--size-coeff",
        type=float,
        default=0.005,
        help="Size regularization coefficient (higher = sparser, default=0.005)",
    )
    parser.add_argument(
        "--contrast-coeff",
        type=float,
        default=0.0,
        help="Contrastive loss coefficient (pushes down other classes, default=1.0)",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    use_gpu = args.gpu and torch.cuda.is_available()
    print(f"Using {'GPU' if use_gpu else 'CPU'}")

    # Load checkpoint
    ckpt = load_checkpoint(args.ckptdir, args.dataset, args.hidden_dim, args.output_dim)
    cg_dict = ckpt["cg"]

    # Extract data from checkpoint
    adj = cg_dict["adj"]  # Adjacency matrices
    feat = cg_dict["feat"]  # Node features
    label = cg_dict["label"]  # Labels
    pred = cg_dict["pred"]  # Model predictions

    input_dim = feat.shape[2]
    num_classes = pred.shape[2]

    print(f"Dataset: {args.dataset}")
    print(f"Input dim: {input_dim}, Num classes: {num_classes}")
    print(f"Adj shape: {adj.shape}, Feat shape: {feat.shape}")

    # Build model
    model = build_model(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_gc_layers,
        use_gpu=use_gpu,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Create explainer
    explainer = GNNExplainer(
        model=model,
        num_hops=args.num_gc_layers,
        lr=args.lr,
        num_epochs=args.num_epochs,
        use_gpu=use_gpu,
    )

    # Set coefficients
    explainer.set_coeffs(size=args.size_coeff, contrast=args.contrast_coeff)
    print(
        f"Size coefficient: {args.size_coeff}, Contrast coefficient: {args.contrast_coeff}"
    )

    # Node classification explanation
    node_idx = args.node_idx
    graph_idx = 0  # Single graph for node classification

    print(f"\nExplaining node {node_idx}")
    print(f"Node label: {label[graph_idx][node_idx]}")
    print(f"Predicted label: {np.argmax(pred[graph_idx][node_idx])}")

    # Extract neighborhood
    node_idx_new, sub_adj, neighbors = extract_neighborhood(
        adj[graph_idx], node_idx, args.num_gc_layers
    )
    sub_feat = feat[graph_idx][neighbors]

    print(f"Subgraph size: {len(neighbors)} nodes")
    print(f"Node index in subgraph: {node_idx_new}")

    adj_tensor = torch.tensor(sub_adj, dtype=torch.float)
    feat_tensor = torch.tensor(sub_feat, dtype=torch.float)

    # Get predicted labels for the subgraph nodes
    sub_pred_label = np.argmax(pred[graph_idx][neighbors], axis=1)

    masked_adj, edge_mask, feat_mask = explainer.explain_node(
        node_idx=node_idx_new,
        adj=adj_tensor,
        features=feat_tensor,
        pred_label=sub_pred_label,
    )

    # Apply top-k filtering if specified
    if args.top_k is not None:
        masked_adj = GNNExplainer.filter_top_k(masked_adj, top_k=args.top_k)
        print(f"Filtered to top {args.top_k} edges")

    print(f"\nExplanation complete!")
    print(f"Masked adj shape: {masked_adj.shape}")
    print(
        f"Non-zero edges in explanation: {np.count_nonzero(masked_adj > args.threshold)}"
    )

    # Show top features
    feat_importance = feat_mask.cpu().numpy()
    top_feats = np.argsort(feat_importance)[-5:][::-1]
    print(f"Top 5 important features: {top_feats}")
    print(f"Feature importance values: {feat_importance[top_feats]}")

    if not args.no_viz:
        visualize_explanation(
            masked_adj,
            node_idx=node_idx_new,
            threshold=args.threshold,
            title=f"Node {node_idx} Explanation",
        )



if __name__ == "__main__":
    main()

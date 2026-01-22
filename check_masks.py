import numpy as np
import os

OUTPUT_DIR = "explanations"
CLASS_NAMES = ['negative', 'neutral', 'positive']

for class_name in CLASS_NAMES:
    node_mask_path = os.path.join(OUTPUT_DIR, f'mean_node_mask_{class_name}.npy')
    edge_mask_path = os.path.join(OUTPUT_DIR, f'mean_edge_mask_{class_name}.npy')

    if not os.path.exists(node_mask_path):
        continue

    node_mask = np.load(node_mask_path)
    edge_mask = np.load(edge_mask_path)

    print(f"\n{class_name.upper()}:")
    print(f"  Node mask shape: {node_mask.shape}")
    print(f"  Node mask range: [{node_mask.min():.6f}, {node_mask.max():.6f}]")
    print(f"  Node mask mean:  {node_mask.mean():.6f}")
    print(f"  Node mask std:   {node_mask.std():.6f}")
    print(f"  Edge mask range: [{edge_mask.min():.6f}, {edge_mask.max():.6f}]")
    print(f"  Edge mask mean:  {edge_mask.mean():.6f}")

    # Check if values are basically uniform
    node_importance = node_mask.mean(axis=1)
    print(f"  Node importance range: [{node_importance.min():.6f}, {node_importance.max():.6f}]")
    print(f"  Difference (max-min): {node_importance.max() - node_importance.min():.6f}")

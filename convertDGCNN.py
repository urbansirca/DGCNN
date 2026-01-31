import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import MessagePassing


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================


class PyGGraphConvolution(MessagePassing):
    """
    Graph convolution using PyG's message passing for GNNExplainer compatibility.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        # Step 1: Propagate first (equivalent to adj @ x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        # Step 2: Then apply weight transformation
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class PyGChebynet(nn.Module):
    """
    Chebyshev graph convolution using PyG message passing.

    Original Chebynet generates Chebyshev polynomial bases:
    - T_0(L) = I (identity)
    - T_1(L) = L (normalized adjacency)
    - T_k(L) = 2*L*T_{k-1}(L) - T_{k-2}(L) (recursive)

    For GNNExplainer compatibility, we use edge_weight to control the adjacency.
    T_0 term: just x @ weight (identity, no graph propagation)
    T_1 term: (L @ x) @ weight (use normalized adjacency)
    """

    def __init__(self, in_channels: int, num_layers: int, out_channels: int):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList(
            [PyGGraphConvolution(in_channels, out_channels) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        self_loop_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges] - includes self-loops
            edge_weight: Normalized edge weights [num_edges] (from D^-0.5 @ A @ D^-0.5)
            num_nodes: Number of nodes
            self_loop_mask: Boolean mask indicating which edges are self-loops [num_edges]
        """
        result = None

        for k in range(self.num_layers):
            if k == 0:
                # T_0(L) * x = I * x
                if self_loop_mask is not None:
                    weights_k = torch.where(
                        self_loop_mask,
                        torch.ones_like(edge_weight),
                        torch.zeros_like(edge_weight),
                    )
                else:
                    weights_k = edge_weight
            else:
                # T_1(L) * x = L * x
                weights_k = edge_weight

            conv_out = self.convs[k](x, edge_index, weights_k)

            if result is None:
                result = conv_out
            else:
                result = result + conv_out

        return F.relu(result)


class DGCNNForExplainerPyG(nn.Module):
    """
    DGCNN rewritten to use PyG message passing for GNNExplainer compatibility.
    """

    def __init__(self, trained_model):
        super().__init__()
        self.in_channels = trained_model.in_channels
        self.num_electrodes = trained_model.num_electrodes
        self.hid_channels = trained_model.hid_channels
        self.num_layers = trained_model.num_layers
        self.num_classes = trained_model.num_classes

        # Create new PyG-compatible layers
        self.layer1 = PyGChebynet(self.in_channels, self.num_layers, self.hid_channels)

        # Copy weights from trained model's Chebynet
        self._copy_chebynet_weights(trained_model.layer1)

        # Copy other layers directly
        self.BN1 = trained_model.BN1
        self.fc1 = trained_model.fc1
        self.fc2 = trained_model.fc2

        # Store learned adjacency
        self.register_buffer("learned_A", trained_model.A.data.clone())

        # Will be set when preparing edge_index
        self.register_buffer("self_loop_mask", None)

    def _copy_chebynet_weights(self, original_chebynet):
        """Copy weights from original Chebynet to PyG version."""
        for new_conv, old_conv in zip(self.layer1.convs, original_chebynet.gc1):
            new_conv.weight.data = old_conv.weight.data.clone()
            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data.clone()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass compatible with GNNExplainer.

        Args:
            x: Node features [num_nodes, in_channels] = [62, 5]
            edge_index: Edge indices [2, num_edges] - includes self-loops
            edge_weight: Edge weights [num_edges] - this is what GNNExplainer perturbs
        """
        # Handle input shape
        if x.dim() == 3:
            x = x.squeeze(0)

        # Apply batch norm
        x_bn = x.unsqueeze(0)
        x_bn = self.BN1(x_bn.transpose(1, 2)).transpose(1, 2)
        x = x_bn.squeeze(0)

        # Use provided edge_weight or compute from learned adjacency
        if edge_weight is None:
            edge_weight = self._get_edge_weights_from_adj(edge_index)

        # Create self-loop mask if not already set
        self_loop_mask = self._get_self_loop_mask(edge_index)

        # Graph convolution
        result = self.layer1(
            x, edge_index, edge_weight, self.num_electrodes, self_loop_mask
        )

        # Flatten and classify
        result = result.reshape(1, -1)
        result = F.relu(self.fc1(result))
        result = self.fc2(result)

        return result

    def _get_self_loop_mask(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Create a mask indicating which edges are self-loops."""
        return edge_index[0] == edge_index[1]

    def _get_edge_weights_from_adj(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Get edge weights from the learned adjacency matrix."""
        A = F.relu(self.learned_A)
        edge_weight = A[edge_index[0], edge_index[1]]
        return edge_weight

    def _get_normalized_edge_weights(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized edge weights matching original normalize_A function.

        Original normalize_A does:
            A = F.relu(A)
            d = torch.sum(A, 1)  # row sum of full matrix
            d = 1 / sqrt(d + 1e-10)
            D = diag(d)
            L = D @ A @ D  (symmetric normalization)

        """
        A = F.relu(self.learned_A)

        # Compute degree from FULL adjacency matrix (row sums)
        deg = A.sum(dim=1)

        # Compute D^-0.5
        deg_inv_sqrt = (deg + 1e-10).pow(-0.5)

        # edge_index: [0]=source=j, [1]=target=i
        # We need weight L[i,j] = deg_inv_sqrt[i] * A[i,j] * deg_inv_sqrt[j]
        src, dst = edge_index  # src=j, dst=i
        # A[i,j] = A[dst, src]
        edge_weight = A[dst, src]
        # L[i,j] = D^-0.5[i] * A[i,j] * D^-0.5[j]
        normalized_weight = deg_inv_sqrt[dst] * edge_weight * deg_inv_sqrt[src]

        return normalized_weight

    def get_edge_index_and_attr(self, threshold: float = 0.0):
        """
        Convert learned adjacency to edge format.

        """
        A = F.relu(self.learned_A)

        # Get all edges above threshold
        if threshold > 0:
            mask = A > threshold
        else:
            mask = A > 1e-10

        # Remove diagonal - we'll add self-loops separately for T_0 handling
        diag_mask = torch.eye(self.num_electrodes, dtype=torch.bool, device=A.device)
        mask = mask & ~diag_mask

        # nonzero gives [row, col] = [i, j] where A[i,j] > threshold
        indices = torch.nonzero(mask, as_tuple=False).t().contiguous()
        # For L @ x, we need edge (j -> i), so SWAP to get [col, row] = [j, i]
        edge_index = torch.stack([indices[1], indices[0]], dim=0)
        # Weight is still A[i,j] = A[indices[0], indices[1]]
        edge_attr = A[indices[0], indices[1]]

        # Add self-loops (needed for T_0 identity operation)
        self_loops = torch.arange(self.num_electrodes, device=A.device)
        self_loop_index = torch.stack([self_loops, self_loops])
        # Self-loop weights set to 1 (identity for T_0)
        self_loop_attr = torch.ones(self.num_electrodes, device=A.device)

        # Concatenate: regular edges + self-loops
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)
        edge_attr = torch.cat([edge_attr, self_loop_attr])

        return edge_index, edge_attr


def prepare_for_explainer_pyg(trained_model, threshold: float = 0.0):
    """Prepare model for GNNExplainer using PyG-compatible version."""
    wrapper = DGCNNForExplainerPyG(trained_model)
    wrapper.eval()

    for param in wrapper.parameters():
        param.requires_grad = False

    edge_index, edge_attr = wrapper.get_edge_index_and_attr(threshold=threshold)

    return wrapper, edge_index, edge_attr



# ============================================================================
# CLASS-LEVEL PROTOTYPE EXPLANATIONS (GNNExplainer Paper Method)
# ============================================================================

def get_embedding(
    model: DGCNNForExplainerPyG,
    sample: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Extract embedding before final classification layer.

    """
    # Handle input shape
    if sample.dim() == 3:
        sample = sample.squeeze(0)

    # Apply batch norm
    x = sample.unsqueeze(0)
    x = model.BN1(x.transpose(1, 2)).transpose(1, 2)
    x = x.squeeze(0)

    # Get self-loop mask
    self_loop_mask = model._get_self_loop_mask(edge_index)

    # Graph convolution
    result = model.layer1(x, edge_index, edge_weight, model.num_electrodes, self_loop_mask)

    # Flatten (this is the "embedding" before fc layers)
    embedding = result.reshape(-1)

    return embedding


def get_class_prototype_explanation(
    model: DGCNNForExplainerPyG,
    explainer,
    data_loader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    target_class: int,
    num_samples: int = 100,
    use_contrastive: bool = False,
    contrast_class: int = None,
    contrast_weight: float = 1.0,
):
    """
    Generate class-level prototype explanation following the GNNExplainer paper.

    This implements the 4-step process described in the original paper:
    1. Find reference instance (embedding closest to class mean)
    2. Compute explanations for many instances in the class
    3. Align explanation graphs to reference
    4. Aggregate with median for robustness to outliers
    """
    from contrastive_explainer import explain_class_contrast

    model.eval()

    print(f"\n{'='*60}")
    print(f"Computing Class Prototype Explanation for Class {target_class}")
    print(f"{'='*60}")

    embeddings = []
    samples = []

    for x, y in data_loader:
        for i in range(len(y)):
            if y[i].item() == target_class:
                sample = x[i].squeeze()
                samples.append(sample)

                # Get embedding (before final classification layer)
                with torch.no_grad():
                    emb = get_embedding(model, sample, edge_index, edge_weight)
                embeddings.append(emb)

                print(f"  Collected {len(embeddings)}/{num_samples} samples", end="\r")

                if len(embeddings) >= num_samples:
                    break
        if len(embeddings) >= num_samples:
            break

    print(f"\n  Collected {len(embeddings)} samples")

    if len(embeddings) == 0:
        raise ValueError(f"No samples found for class {target_class}")

    # Compute mean embedding
    embeddings = torch.stack(embeddings)
    mean_embedding = embeddings.mean(dim=0)

    # Find reference instance (closest to mean)
    distances = torch.norm(embeddings - mean_embedding, dim=1)
    reference_idx = distances.argmin().item()
    reference_sample = samples[reference_idx]
    reference_embedding = embeddings[reference_idx]

    print(f"  Reference index: {reference_idx}")
    print(f"  Distance to mean: {distances[reference_idx]:.4f}")
    print(f"  Mean distance: {distances.mean():.4f}")
    print(f"  Max distance: {distances.max():.4f}")

    #  Compute explanations for all samples

    node_masks = []
    edge_masks = []

    for idx, sample in enumerate(samples):
        if use_contrastive and contrast_class is not None:
            explanation = explain_class_contrast(
                explainer_model=model,
                sample=sample,
                edge_index=edge_index,
                edge_weight=edge_weight,
                target_class=target_class,
                contrast_class=contrast_class,
                epochs=200,
                contrast_weight=contrast_weight,
            )
        else:
            explanation = explainer(
                x=sample,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

        node_masks.append(explanation.node_mask.detach().cpu())
        edge_masks.append(explanation.edge_mask.detach().cpu())

        print(f"  Computed {idx + 1}/{len(samples)} explanations", end="\r")

    print(f"\n  Completed {len(samples)} explanations")

    #  Aggregate with MEDIAN (robust to outliers)
    print(f"\nStep 4: Aggregating explanations with median...")

    node_masks = torch.stack(node_masks)
    edge_masks = torch.stack(edge_masks)

    # Median aggregation (robust to outlier explanations)
    prototype_node_mask = torch.median(node_masks, dim=0).values
    prototype_edge_mask = torch.median(edge_masks, dim=0).values

    # Also compute mean for comparison
    mean_node_mask = node_masks.mean(dim=0)
    mean_edge_mask = edge_masks.mean(dim=0)

    # Compute standard deviation for uncertainty quantification
    std_node_mask = node_masks.std(dim=0)
    std_edge_mask = edge_masks.std(dim=0)

    print(f"  Prototype node mask shape: {prototype_node_mask.shape}")
    print(f"  Prototype edge mask shape: {prototype_edge_mask.shape}")

    # Compute agreement statistics
    node_agreement = 1 - (std_node_mask / (mean_node_mask.abs() + 1e-10)).mean()
    edge_agreement = 1 - (std_edge_mask / (mean_edge_mask.abs() + 1e-10)).mean()
    print(f"  Node mask agreement (1 - CV): {node_agreement:.4f}")
    print(f"  Edge mask agreement (1 - CV): {edge_agreement:.4f}")

    return {
        'prototype_node_mask': prototype_node_mask,
        'prototype_edge_mask': prototype_edge_mask,
        'reference_sample': reference_sample,
        'reference_idx': reference_idx,
        'reference_embedding': reference_embedding,
        'mean_embedding': mean_embedding,
        'num_samples': len(samples),
        'mean_node_mask': mean_node_mask,
        'mean_edge_mask': mean_edge_mask,
        'std_node_mask': std_node_mask,
        'std_edge_mask': std_edge_mask,
        'all_node_masks': node_masks,
        'all_edge_masks': edge_masks,
        'embedding_distances': distances,
    }


def get_all_class_prototypes(
    model: DGCNNForExplainerPyG,
    explainer,
    data_loader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_classes: int = 3,
    num_samples_per_class: int = 100,
    use_contrastive: bool = False,
    contrast_weight: float = 1.0,
):
    """Compute class prototype explanations for all classes."""
    prototypes = {}

    for class_idx in range(num_classes):
        # For contrastive, use the "most different" class as contrast
        if use_contrastive:
            if class_idx == 0:  # Negative
                contrast_class = 2  # Positive
            elif class_idx == 2:  # Positive
                contrast_class = 0  # Negative
            else:  # Neutral
                contrast_class = 0  # Contrast with Negative
        else:
            contrast_class = None

        prototype = get_class_prototype_explanation(
            model=model,
            explainer=explainer,
            data_loader=data_loader,
            edge_index=edge_index,
            edge_weight=edge_weight,
            target_class=class_idx,
            num_samples=num_samples_per_class,
            use_contrastive=use_contrastive,
            contrast_class=contrast_class,
            contrast_weight=contrast_weight,
        )
        prototypes[class_idx] = prototype

    return prototypes


# ============================================================================
# VALIDATION METRICS
# ============================================================================

def compute_fidelity_plus(
    model,
    samples: list,
    edge_masks: list,
    edge_index: torch.Tensor,
    top_k_percentile: int = 20,
    device: str = 'cpu',
):
    """
    Fidelity+ (Sufficiency): Remove top-k% important edges and measure accuracy DROP.

    High Fidelity+ means: removing important edges significantly hurts accuracy.
    This validates that the explanation identifies truly important edges.
    """
    model.eval()

    original_correct = 0
    masked_correct = 0
    total = len(samples)

    # DEBUG: Track prediction changes
    prediction_changes = 0
    correct_to_incorrect = 0

    for i, (x, y) in enumerate(samples):
        if isinstance(x, torch.Tensor):
            x = x.squeeze()
        else:
            x = torch.tensor(x).squeeze()

        edge_mask = edge_masks[i]
        if isinstance(edge_mask, torch.Tensor):
            edge_mask = edge_mask.cpu().numpy()

        # Original prediction
        with torch.no_grad():
            edge_weight = model._get_normalized_edge_weights(edge_index)
            pred_orig = model(x, edge_index, edge_weight).argmax(dim=1).item()
            if pred_orig == y:
                original_correct += 1

        # Find threshold for top-k% edges
        threshold = np.percentile(edge_mask.flatten(), 100 - top_k_percentile)

        # Create masked edge weights (remove important edges)
        mask_tensor = torch.tensor(edge_mask >= threshold, device=edge_weight.device)
        masked_edge_weight = edge_weight.clone()
        masked_edge_weight[mask_tensor] = 0

        # Masked prediction
        with torch.no_grad():
            pred_masked = model(x, edge_index, masked_edge_weight).argmax(dim=1).item()
            if pred_masked == y:
                masked_correct += 1

            # Track changes
            if pred_masked != pred_orig:
                prediction_changes += 1
            if pred_orig == y and pred_masked != y:
                correct_to_incorrect += 1

    original_acc = original_correct / total
    masked_acc = masked_correct / total
    fidelity_plus = original_acc - masked_acc

    return {
        'original_accuracy': original_acc,
        'masked_accuracy': masked_acc,
        'fidelity_plus': fidelity_plus,
        'top_k_percentile': top_k_percentile,
        'prediction_changes': prediction_changes,
        'correct_to_incorrect': correct_to_incorrect,
        'total_samples': total
    }


def compute_fidelity_minus(
    model,
    samples: list,
    edge_masks: list,
    edge_index: torch.Tensor,
    top_k_percentile: int = 20,
    device: str = 'cpu',
):
    """
    Fidelity- (Comprehensiveness): Keep ONLY top-k% important edges and measure accuracy.

    High Fidelity- means: keeping only important edges maintains accuracy.
    This validates that the explanation captures sufficient information.
    """
    model.eval()

    original_correct = 0
    sparse_correct = 0
    total = len(samples)

    for i, (x, y) in enumerate(samples):
        if isinstance(x, torch.Tensor):
            x = x.squeeze()
        else:
            x = torch.tensor(x).squeeze()

        edge_mask = edge_masks[i]
        if isinstance(edge_mask, torch.Tensor):
            edge_mask = edge_mask.cpu().numpy()

        # Original prediction
        with torch.no_grad():
            edge_weight = model._get_normalized_edge_weights(edge_index)
            pred_orig = model(x, edge_index, edge_weight).argmax(dim=1).item()
            if pred_orig == y:
                original_correct += 1

        # Find threshold for top-k% edges
        threshold = np.percentile(edge_mask.flatten(), 100 - top_k_percentile)

        # Create sparse edge weights (keep only important edges)
        mask_tensor = torch.tensor(edge_mask >= threshold, device=edge_weight.device)
        sparse_edge_weight = torch.zeros_like(edge_weight)
        sparse_edge_weight[mask_tensor] = edge_weight[mask_tensor]

        # Sparse prediction
        with torch.no_grad():
            pred_sparse = model(x, edge_index, sparse_edge_weight).argmax(dim=1).item()
            if pred_sparse == y:
                sparse_correct += 1

    original_acc = original_correct / total
    sparse_acc = sparse_correct / total
    fidelity_minus = sparse_acc

    return {
        'original_accuracy': original_acc,
        'sparse_accuracy': sparse_acc,
        'fidelity_minus': fidelity_minus,
        'top_k_percentile': top_k_percentile
    }


def compute_sparsity_curve(
    model,
    samples: list,
    edge_masks: list,
    edge_index: torch.Tensor,
    percentiles: list = None,
    device: str = 'cpu',
):
    """
    Compute accuracy vs sparsity curve.

    Shows how accuracy changes as we keep fewer edges (only the most important ones).
    A good explanation should maintain high accuracy with few edges.
    """
    if percentiles is None:
        percentiles = [5, 10, 20, 30, 50, 70, 100]

    results = {'percentiles': percentiles, 'accuracies': []}

    for pct in percentiles:
        fidelity_result = compute_fidelity_minus(model, samples, edge_masks, edge_index,
                                                  top_k_percentile=pct, device=device)
        results['accuracies'].append(fidelity_result['sparse_accuracy'])

    return results


def compute_stability(edge_masks_list: list):
    """
    Compute stability/consistency of explanations across multiple samples.

    High stability means the explainer gives consistent results for similar inputs.
    Uses Intersection over Union (IoU) of top-k edges.
    """
    n_samples = len(edge_masks_list)
    if n_samples < 2:
        return {'mean_iou': 1.0, 'std_iou': 0.0, 'n_comparisons': 0}

    ious = []
    top_k_percent = 10

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            mask_i = edge_masks_list[i]
            mask_j = edge_masks_list[j]

            if isinstance(mask_i, torch.Tensor):
                mask_i = mask_i.cpu().numpy()
            if isinstance(mask_j, torch.Tensor):
                mask_j = mask_j.cpu().numpy()

            mask_i = mask_i.flatten()
            mask_j = mask_j.flatten()

            threshold_i = np.percentile(mask_i, 100 - top_k_percent)
            threshold_j = np.percentile(mask_j, 100 - top_k_percent)

            top_i = set(np.where(mask_i >= threshold_i)[0])
            top_j = set(np.where(mask_j >= threshold_j)[0])

            intersection = len(top_i & top_j)
            union = len(top_i | top_j)
            iou = intersection / union if union > 0 else 0
            ious.append(iou)

    return {
        'mean_iou': np.mean(ious),
        'std_iou': np.std(ious),
        'n_comparisons': len(ious)
    }


def compute_all_validation_metrics(
    model,
    samples: list,
    edge_masks: list,
    edge_index: torch.Tensor,
    device: str = 'cpu',
):
    """Compute all validation metrics for explanation results."""
    print("\nComputing validation metrics...")

    print("  Computing Fidelity+ (sufficiency)...")
    fidelity_plus = compute_fidelity_plus(model, samples, edge_masks, edge_index,
                                           top_k_percentile=40, device=device)

    print("  Computing Fidelity- (comprehensiveness)...")
    fidelity_minus = compute_fidelity_minus(model, samples, edge_masks, edge_index,
                                             top_k_percentile=40, device=device)

    print("  Computing sparsity curve...")
    sparsity = compute_sparsity_curve(model, samples, edge_masks, edge_index, device=device)

    print("  Computing stability...")
    stability = compute_stability(edge_masks)

    return {
        'fidelity_plus': fidelity_plus,
        'fidelity_minus': fidelity_minus,
        'sparsity_curve': sparsity,
        'stability': stability
    }


def print_validation_metrics(metrics: dict, name: str):
    """Print validation metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"VALIDATION METRICS - {name}")
    print(f"{'='*60}")

    fp = metrics['fidelity_plus']
    print(f"\nFidelity+ (Sufficiency) - Removing top {fp['top_k_percentile']}% edges:")
    print(f"  Original accuracy: {fp['original_accuracy']:.4f}")
    print(f"  After removal:     {fp['masked_accuracy']:.4f}")
    print(f"  Fidelity+ score:   {fp['fidelity_plus']:.4f} (higher = better explanation)")
    if 'prediction_changes' in fp:
        print(f"  Prediction changes: {fp['prediction_changes']}/{fp['total_samples']}")
        print(f"  Correct->Incorrect: {fp['correct_to_incorrect']}/{fp['total_samples']}")

    fm = metrics['fidelity_minus']
    print(f"\nFidelity- (Comprehensiveness) - Keeping only top {fm['top_k_percentile']}% edges:")
    print(f"  Original accuracy: {fm['original_accuracy']:.4f}")
    print(f"  Sparse accuracy:   {fm['sparse_accuracy']:.4f}")
    print(f"  Fidelity- score:   {fm['fidelity_minus']:.4f} (higher = better explanation)")

    stab = metrics['stability']
    print(f"\nStability (IoU of top edges across samples):")
    print(f"  Mean IoU:          {stab['mean_iou']:.4f}")
    print(f"  Std IoU:           {stab['std_iou']:.4f}")
    print(f"  Comparisons:       {stab['n_comparisons']}")

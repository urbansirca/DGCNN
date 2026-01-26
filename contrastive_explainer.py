from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelMode
import torch
import torch.nn.functional as F


class ContrastiveGNNExplainer(GNNExplainer):
    """
    GNNExplainer with contrastive loss to find features that distinguish
    one class from another.

    Loss: -log P(Y=c | G_s) + log P(Y=c' | G_s) + λ1 * ||M_e||_1 + λ2 * H(M_e)

    Args:
        epochs (int): Number of optimization epochs
        lr (float): Learning rate
        contrast_class (int): The contrasting class c' to push away from
        contrast_weight (float): Weight for the contrastive term (default: 1.0)
        **kwargs: Additional coefficients passed to GNNExplainer
    """

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        contrast_class: int = None,
        contrast_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(epochs=epochs, lr=lr, **kwargs)
        self.contrast_class = contrast_class
        self.contrast_weight = contrast_weight

    def _loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss that maximizes probability of target class
        while minimizing probability of contrast class.
        """
        if self.contrast_class is None:
            # Fall back to standard loss if no contrast class specified
            return super()._loss(y_hat, y)

        # Calculate contrastive loss
        loss = self._calculate_contrastive_loss(y_hat, y)

        # Apply regularization (same as parent class)
        if self.is_hetero:
            loss = self._apply_hetero_regularization(loss)
        else:
            loss = self._apply_homo_regularization(loss)

        return loss

    def _calculate_contrastive_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute: -log P(Y=c | G_s) + contrast_weight * log P(Y=c' | G_s)

        This encourages the subgraph to:
        1. Maximize probability of the true class c
        2. Minimize probability of the contrast class c'
        """
        # Get log probabilities
        log_probs = F.log_softmax(y_hat, dim=-1)

        # Handle different input shapes
        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(0)
            log_probs = F.log_softmax(y_hat, dim=-1)

        if y.dim() == 0:
            y = y.unsqueeze(0)

        # Term 1: -log P(Y=c | G_s) - maximize true class probability
        target_class = y[0].item() if y.numel() == 1 else y
        nll_target = F.nll_loss(log_probs, y)

        # Term 2: +log P(Y=c' | G_s) - minimize contrast class probability
        contrast_target = torch.tensor(
            [self.contrast_class], device=y_hat.device, dtype=torch.long
        )

        if log_probs.size(0) > 1:
            contrast_target = contrast_target.expand(log_probs.size(0))

        # Note: We ADD this term (not subtract) because we want to MAXIMIZE
        # P(Y=c') in the negative direction, i.e., minimize it
        log_prob_contrast = F.nll_loss(log_probs, contrast_target)

        # Contrastive loss: -log P(c) + weight * log P(c')
        # Since nll_loss = -log P, we have:
        # loss = nll_target - contrast_weight * log_prob_contrast
        # which equals: -log P(c) - contrast_weight * (-log P(c'))
        # which equals: -log P(c) + contrast_weight * log P(c')
        loss = nll_target - self.contrast_weight * log_prob_contrast

        return loss


def explain_class_contrast(
    explainer_model,
    sample: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    target_class: int,
    contrast_class: int,
    epochs: int = 200,
    contrast_weight: float = 1.0,
    **explainer_kwargs
):
    """
    Convenience function to explain what distinguishes target_class from contrast_class.

    Args:
        explainer_model: The PyG-compatible model
        sample: Input features [num_nodes, num_features]
        edge_index: Edge indices [2, num_edges]
        edge_weight: Edge weights [num_edges]
        target_class: The class we want to explain (c)
        contrast_class: The class we want to contrast against (c')
        epochs: Number of optimization epochs
        contrast_weight: Weight for the contrastive term
        **explainer_kwargs: Additional kwargs for the explainer

    Returns:
        Explanation object with node_mask and edge_mask
    """
    from torch_geometric.explain import Explainer

    explainer = Explainer(
        model=explainer_model,
        algorithm=ContrastiveGNNExplainer(
            epochs=epochs,
            contrast_class=contrast_class,
            contrast_weight=contrast_weight,
            **explainer_kwargs
        ),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    explanation = explainer(
        x=sample,
        edge_index=edge_index,
        edge_weight=edge_weight,
        target=torch.tensor([target_class]),
    )

    return explanation

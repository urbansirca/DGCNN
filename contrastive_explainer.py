from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelMode
import torch
import torch.nn.functional as F


class ContrastiveGNNExplainer(GNNExplainer):
    """
    GNNExplainer with contrastive loss to find features that distinguish
    one class from another class or all other classes.

    Loss (one-vs-one): -log P(Y=c | G_s) + λ_contrast * log P(Y=c' | G_s)
          + λ1 * ||M_e||_1 + λ2 * H(M_e)

    Loss (one-vs-rest): -log P(Y=c | G_s) + λ_contrast * mean(log P(Y=c' | G_s)) for all c' != c
          + λ1 * ||M_e||_1 + λ2 * H(M_e)

    Args:
        epochs (int): Number of optimization epochs
        lr (float): Learning rate
        num_classes (int): Total number of classes
        contrast_weight (float): Weight for the contrastive term (default: 1.0)
        contrast_class (int, optional): Specific class to contrast against (one-vs-one).
                                        If None, contrast against all other classes (one-vs-rest).
        **kwargs: Additional coefficients passed to GNNExplainer
    """

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        num_classes: int = 3,
        contrast_weight: float = 1.0,
        contrast_class: int = None,
        **kwargs
    ):
        super().__init__(epochs=epochs, lr=lr, **kwargs)
        self.num_classes = num_classes
        self.contrast_weight = contrast_weight
        self.contrast_class = contrast_class

    def _loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss that maximizes probability of target class
        while minimizing probability of all other classes.
        """
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
        Compute contrastive loss based on mode:

        One-vs-one: -log P(Y=c | G_s) + contrast_weight * log P(Y=c' | G_s)
        One-vs-rest: -log P(Y=c | G_s) + contrast_weight * mean(log P(Y=c' | G_s)) for all c' != c
        """
        # Handle different input shapes
        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(0)

        if y.dim() == 0:
            y = y.unsqueeze(0)

        # Get log probabilities
        log_probs = F.log_softmax(y_hat, dim=-1)

        # Term 1: -log P(Y=c | G_s) - maximize true class probability
        target_class = y[0].item()
        nll_target = F.nll_loss(log_probs, y)

        # Term 2: Contrastive term - depends on mode
        if self.contrast_class is not None:
            # One-vs-one: contrast against specific class
            contrast_log_prob = log_probs[:, self.contrast_class]
            # We want to minimize P(c'), so we subtract the log prob
            loss = nll_target - self.contrast_weight * contrast_log_prob.mean()
        else:
            # One-vs-rest: contrast against all other classes
            other_classes = [c for c in range(self.num_classes) if c != target_class]

            if len(other_classes) > 0:
                # Get log probabilities for all other classes
                other_log_probs = log_probs[:, other_classes]  # [batch, num_other_classes]
                # Average log probability across other classes
                mean_other_log_prob = other_log_probs.mean()

                # Contrastive loss: -log P(c) - contrast_weight * mean(log P(c'))
                loss = nll_target - self.contrast_weight * mean_other_log_prob
            else:
                loss = nll_target

        return loss


def explain_class_contrast(
    explainer_model,
    sample: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    target_class: int,
    contrast_class: int = None,
    num_classes: int = 3,
    epochs: int = 200,
    contrast_weight: float = 1.0,
    **explainer_kwargs
):
    """
    Convenience function to explain what distinguishes target_class from contrast_class
    or all other classes.

    Args:
        explainer_model: The PyG-compatible model
        sample: Input features [num_nodes, num_features]
        edge_index: Edge indices [2, num_edges]
        edge_weight: Edge weights [num_edges]
        target_class: The class we want to explain (c)
        contrast_class: The specific class to contrast against (c').
                       If None, contrast against all other classes (one-vs-rest).
        num_classes: Total number of classes
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
            num_classes=num_classes,
            contrast_weight=contrast_weight,
            contrast_class=contrast_class,
            **explainer_kwargs
        ),
        explanation_type="phenomenon",
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

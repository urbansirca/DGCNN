import math
import torch
import torch.nn as nn
import numpy as np


class ExplainModule(nn.Module):
    """
    Internal module that learns edge and feature masks for node classification.
    """

    def __init__(
        self,
        adj,
        x,
        model,
        label,
        graph_idx=0,
        mask_act="sigmoid",
        use_gpu=False,
    ):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.mask_act = mask_act
        self.use_gpu = use_gpu

        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self._construct_edge_mask(num_nodes)
        self.feat_mask = self._construct_feat_mask(x.size(-1))

        # Diagonal mask (no self-loops)
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if use_gpu:
            self.diag_mask = self.diag_mask.cuda()

        # Optimizer
        params = [self.mask, self.feat_mask]
        self.optimizer = torch.optim.Adam(params, lr=0.1)

        # Loss coefficients
        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "lap": 1.0,
            "contrast": 1.0,  # Contrastive loss weight (pushes down other classes)
        }

    def _construct_edge_mask(self, num_nodes):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        std = nn.init.calculate_gain("relu") * math.sqrt(2.0 / (num_nodes + num_nodes))
        with torch.no_grad():
            mask.normal_(1.0, std)
        if self.use_gpu:
            mask = nn.Parameter(mask.cuda())

        mask_bias = None  # Not using bias by default
        return mask, mask_bias

    def _construct_feat_mask(self, feat_dim):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        with torch.no_grad():
            nn.init.constant_(mask, 0.0)
        if self.use_gpu:
            mask = nn.Parameter(mask.cuda())
        return mask

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.use_gpu else self.adj
        masked_adj = adj * sym_mask
        return masked_adj * self.diag_mask

    def forward(self, node_idx):
        x = self.x.cuda() if self.use_gpu else self.x

        self.masked_adj = self._masked_adj()

        # Apply feature mask
        feat_mask = torch.sigmoid(self.feat_mask)
        x = x * feat_mask

        ypred, adj_att = self.model(x, self.masked_adj)

        node_pred = ypred[self.graph_idx, node_idx, :]
        res = nn.Softmax(dim=0)(node_pred)

        return res, adj_att

    def loss(self, pred, pred_label, node_idx, epoch):
        eps = 1e-8

        # Get target class
        target_class = int(pred_label[node_idx])
        num_classes = pred.shape[0]

        # Base prediction loss
        p_target = torch.clamp(pred[target_class], min=eps, max=1.0)
        base_pred_loss = -torch.log(p_target)

        # Contrastive loss: push down other classes (helps find complete explanations)
        contrast_weight = self.coeffs.get("contrast", 1.0)
        if contrast_weight > 0 and num_classes > 1:
            contrast_loss = 0.0
            for c in range(num_classes):
                if c != target_class:
                    p_c = torch.clamp(pred[c], min=eps, max=1.0)
                    contrast_loss += torch.log(p_c)
            contrast_loss = contrast_loss / (num_classes - 1)
            pred_loss = base_pred_loss + contrast_weight * contrast_loss
        else:
            pred_loss = base_pred_loss

        # Edge mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        else:
            mask = self.mask

        # Size loss
        size_loss = self.coeffs["size"] * torch.sum(mask)

        # Entropy loss
        mask_clamped = torch.clamp(mask, eps, 1 - eps)
        mask_ent = -mask_clamped * torch.log(mask_clamped) - (
            1 - mask_clamped
        ) * torch.log(1 - mask_clamped)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        # Feature mask loss
        feat_mask = torch.sigmoid(self.feat_mask)
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # Laplacian regularization
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj[self.graph_idx]
        L = D - m_adj

        pred_label_t = torch.tensor(pred_label, dtype=torch.float, device=L.device)
        lap_loss = (
            self.coeffs["lap"]
            * (pred_label_t @ L @ pred_label_t)
            / self.adj.numel()
        )

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        return loss


class GNNExplainer:
    """
    GNNExplainer learns edge and feature masks to explain GNN predictions for node classification.
    """

    def __init__(
        self,
        model,
        num_hops=3,
        lr=0.1,
        num_epochs=100,
        mask_activation="sigmoid",
        use_gpu=True,
    ):
        self.model = model
        self.model.eval()
        self.num_hops = num_hops
        self.lr = lr
        self.num_epochs = num_epochs
        self.mask_activation = mask_activation
        self.use_gpu = use_gpu and torch.cuda.is_available()

        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "lap": 1.0,
            "contrast": 1.0,  # Contrastive loss weight
        }

    def explain_node(self, node_idx, adj, features, pred_label=None):
        """Explain the prediction for a single node."""
        # Prepare inputs - must have batch dimension
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if features.dim() == 2:
            features = features.unsqueeze(0)

        # Features need requires_grad=True (like original)
        adj = adj.float()
        x = features.float().requires_grad_(True)

        if self.use_gpu:
            adj = adj.cuda()
            x = x.cuda()

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            orig_pred, _ = self.model(x, adj)
            if pred_label is None:
                pred_label = orig_pred[0].argmax(dim=1).cpu().numpy()

        # Create ExplainModule
        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=None,
            graph_idx=0,
            mask_act=self.mask_activation,
            use_gpu=self.use_gpu,
        )

        if self.use_gpu:
            explainer = explainer.cuda()

        # Set coefficients
        explainer.coeffs = self.coeffs.copy()
        explainer.optimizer = torch.optim.Adam(
            [explainer.mask, explainer.feat_mask], lr=self.lr
        )

        # Training loop
        self.model.eval()
        explainer.train()

        for epoch in range(self.num_epochs):
            explainer.zero_grad()
            explainer.optimizer.zero_grad()

            ypred, _ = explainer(node_idx)
            loss = explainer.loss(ypred, pred_label, node_idx, epoch)
            loss.backward()

            explainer.optimizer.step()

            if epoch % 50 == 0:
                print(
                    f"  Epoch {epoch}: loss={loss.item():.4f}, pred={ypred[pred_label[node_idx]].item():.4f}"
                )

        # Get final masked adjacency
        masked_adj = (
            explainer.masked_adj[0].cpu().detach().numpy() * adj[0].cpu().numpy()
        )

        return (
            masked_adj,
            explainer.mask.detach(),
            torch.sigmoid(explainer.feat_mask).detach(),
        )

    def set_coeffs(self, **kwargs):
        """Update loss coefficients."""
        for key, val in kwargs.items():
            if key in self.coeffs:
                self.coeffs[key] = val

    @staticmethod
    def filter_top_k(masked_adj, top_k=10):
        """Keep only the top-k highest weighted edges."""
        adj = masked_adj.copy()
        upper = np.triu(adj, k=1)
        flat = upper.flatten()

        if top_k >= np.count_nonzero(flat):
            return adj

        top_indices = np.argpartition(flat, -top_k)[-top_k:]
        threshold = flat[top_indices].min()

        adj[adj < threshold] = 0
        return adj

    @staticmethod
    def filter_threshold(masked_adj, threshold=0.1):
        """Keep only edges above a threshold."""
        adj = masked_adj.copy()
        adj[adj < threshold] = 0
        return adj


def neighborhoods(adj, n_hops, use_cuda=False):
    """Compute n-hop neighborhood matrix using matrix multiplication."""
    adj = torch.tensor(adj, dtype=torch.float)
    if use_cuda:
        adj = adj.cuda()

    hop_adj = power_adj = adj
    for _ in range(n_hops - 1):
        power_adj = power_adj @ adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()

    return hop_adj.cpu().numpy().astype(int)


def extract_neighborhood(adj, node_idx, num_hops, use_cuda=False):
    """Extract k-hop neighborhood subgraph around a node."""
    if isinstance(adj, torch.Tensor):
        adj_np = adj.cpu().numpy()
    else:
        adj_np = adj

    hop_adj = neighborhoods(adj_np, num_hops, use_cuda)
    neighbors_row = hop_adj[node_idx, :]
    neighbors = np.nonzero(neighbors_row)[0]
    node_idx_new = np.sum(neighbors_row[:node_idx])
    sub_adj = adj_np[np.ix_(neighbors, neighbors)]

    return int(node_idx_new), sub_adj, neighbors

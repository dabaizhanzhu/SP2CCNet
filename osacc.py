import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_cos_sim(x1, x2):
    """
    Calculate pair-wise cosine similarity.

    Args:
        x1: Tensor with shape [..., M, D] or [M, D].
        x2: Tensor with shape [..., N, D].

    Returns:
        Tensor with shape [..., M, N].
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return torch.matmul(x1, x2.transpose(-2, -1))


def orthogonal_prior_generator(num_prototypes, feature_dim):
    """
    Generate L2-normalized orthogonal prototype vectors.

    Args:
        num_prototypes: Number of semantic axes / cluster centers.
        feature_dim: Feature dimension of each prototype.

    Returns:
        nn.Parameter with shape [num_prototypes, feature_dim].
    """
    if num_prototypes <= 0:
        raise ValueError("num_prototypes must be positive.")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")

    weight = torch.randn(num_prototypes, feature_dim) * (2.0 / feature_dim) ** 0.5
    basis = [weight[0]]

    for vector in weight[1:]:
        projection = torch.zeros_like(vector)
        for base in basis:
            denom = torch.matmul(base, base).clamp_min(1e-12)
            projection = projection + torch.matmul(vector, base) / denom * base
        basis.append(vector - projection)

    basis = torch.stack(basis)
    basis = F.normalize(basis, dim=1)
    return nn.Parameter(basis.float())


class OSACC(nn.Module):
    """
    Orthogonal Semantic Axis Context Cluster module.

    Forward flow:
        x -> 1x1 query/value projection -> multi-head reshape
          -> cosine similarity with orthogonal centers
          -> hard semantic-axis assignment
          -> semantic aggregation -> context dispatch
          -> restore spatial shape -> 1x1 output projection

    Args:
        in_channel: Input channels.
        out_channel: Output channels.
        proposal_h: Number of prototype rows, total prototypes = proposal_h * proposal_w.
        proposal_w: Number of prototype columns.
        heads: Number of heads.
        head_dim: Channel dimension of each head.
        use_pooled_value_center: Keep the original OSAWaveNet pooled value-center residual
            in the aggregation step.
        return_legacy_numpy_map: If True, return only the first head assignment as numpy,
            matching the old ContextClusterModule behavior.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        proposal_h=2,
        proposal_w=2,
        heads=4,
        head_dim=24,
        use_pooled_value_center=True,
        return_legacy_numpy_map=False,
    ):
        super().__init__()
        if heads <= 0:
            raise ValueError("heads must be positive.")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive.")

        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.num_prototypes = int(proposal_h) * int(proposal_w)
        self.use_pooled_value_center = bool(use_pooled_value_center)
        self.return_legacy_numpy_map = bool(return_legacy_numpy_map)

        self.f = nn.Conv2d(in_channel, self.heads * self.head_dim, kernel_size=1)
        self.v = nn.Conv2d(in_channel, self.heads * self.head_dim, kernel_size=1)
        self.proj = nn.Conv2d(self.heads * self.head_dim, out_channel, kernel_size=1)

        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers = orthogonal_prior_generator(self.num_prototypes, self.head_dim)
        self.centers.requires_grad = False
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_h, proposal_w))

    def _split_heads(self, x):
        b, _, h, w = x.shape
        x = x.reshape(b, self.heads, self.head_dim, h, w)
        return x.reshape(b * self.heads, self.head_dim, h, w)

    def _merge_heads(self, x, batch_size):
        _, _, h, w = x.shape
        x = x.reshape(batch_size, self.heads, self.head_dim, h, w)
        return x.reshape(batch_size, self.heads * self.head_dim, h, w)

    def forward(self, x):
        """
        Args:
            x: Input feature map with shape [B, C, H, W].

        Returns:
            out: Tensor with shape [B, out_channel, H, W].
            assignment_map: Long tensor with shape [B, heads, H, W]. If
                return_legacy_numpy_map=True, returns the first head map as numpy [H, W].
        """
        batch_size, _, height, width = x.shape

        q = self._split_heads(self.f(x))
        value = self._split_heads(self.v(x))
        batch_heads, channels, _, _ = q.shape

        q_flat = q.reshape(batch_heads, channels, -1).transpose(1, 2)
        sim = torch.sigmoid(
            self.sim_beta + self.sim_alpha * pairwise_cos_sim(self.centers, q_flat)
        )

        sim_max_idx = sim.argmax(dim=1, keepdim=True)
        assignment_map = sim_max_idx.reshape(batch_size, self.heads, height, width)

        hard_mask = torch.zeros_like(sim)
        hard_mask.scatter_(1, sim_max_idx, 1.0)
        sim = sim * hard_mask

        value_flat = value.reshape(batch_heads, channels, -1).transpose(1, 2)
        semantic_context = (value_flat.unsqueeze(1) * sim.unsqueeze(-1)).sum(dim=2)
        semantic_mass = sim.sum(dim=-1, keepdim=True)

        if self.use_pooled_value_center:
            value_centers = self.centers_proposal(value).flatten(2).transpose(1, 2)
            semantic_context = semantic_context + value_centers
            semantic_mass = semantic_mass + 1.0

        semantic_context = semantic_context / semantic_mass.clamp_min(1e-6)
        dispatched = (semantic_context.unsqueeze(2) * sim.unsqueeze(-1)).sum(dim=1)
        out = dispatched.transpose(1, 2).reshape(batch_heads, channels, height, width)
        out = self._merge_heads(out, batch_size)
        out = self.proj(out)

        if self.return_legacy_numpy_map:
            assignment_map = assignment_map[0, 0].detach().cpu().numpy()

        return out, assignment_map


class ContextClusterModule(OSACC):
    """Backward-compatible name for the original OSAWaveNet module."""


OrthogonalPriorGenerator = orthogonal_prior_generator


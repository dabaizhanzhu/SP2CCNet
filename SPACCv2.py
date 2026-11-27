"""
Standalone SPACCv2 module.

SPACCv2 = Semantic Prototype Axes-guided Context Clustering with
ProtoSeg-style CE + PPC + PPD feature-space supervision.
This file contains only the semantic prototype-axis clustering logic.
The original fixed orthogonal-axis OSACC logic is saved separately in osacc.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def pairwise_cos_sim(x1: Tensor, x2: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Compute pair-wise cosine similarity.

    Args:
        x1: Tensor with shape [..., M, D] or [M, D].
        x2: Tensor with shape [..., N, D].
        eps: Numerical epsilon used by normalize.

    Returns:
        Similarity tensor with shape [..., M, N].
    """
    x1 = F.normalize(x1, dim=-1, eps=eps)
    x2 = F.normalize(x2, dim=-1, eps=eps)
    return torch.matmul(x1, x2.transpose(-2, -1))


def distributed_sinkhorn(out: Tensor, sinkhorn_iterations: int = 3, epsilon: float = 0.05) -> Tuple[Tensor, Tensor]:
    """
    ProtoSeg-style Sinkhorn assignment inside one semantic class.

    Args:
        out: Assignment logits with shape [num_pixels, prototypes_per_class].
        sinkhorn_iterations: Number of normalization iterations.
        epsilon: Temperature used before Sinkhorn normalization.

    Returns:
        Hard assignment matrix [num_pixels, prototypes_per_class] and prototype indices.
    """
    q = torch.exp(out / epsilon).t()
    num_pixels = q.shape[1]
    num_prototypes = q.shape[0]

    q /= torch.sum(q).clamp_min(1e-12)
    for _ in range(sinkhorn_iterations):
        q /= torch.sum(q, dim=1, keepdim=True).clamp_min(1e-12)
        q /= num_prototypes

        q /= torch.sum(q, dim=0, keepdim=True).clamp_min(1e-12)
        q /= num_pixels

    q *= num_pixels
    q = q.t()

    indices = torch.argmax(q, dim=1)
    q = F.gumbel_softmax(q, tau=0.5, hard=True)
    return q, indices


def split_heads_2d(x: Tensor, heads: int) -> Tensor:
    """[B, heads*D, H, W] -> [B*heads, D, H, W]."""
    b, channels, h, w = x.shape
    if channels % heads != 0:
        raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
    head_dim = channels // heads
    return x.view(b, heads, head_dim, h, w).reshape(b * heads, head_dim, h, w)


def merge_heads_2d(x: Tensor, heads: int) -> Tensor:
    """[B*heads, D, H, W] -> [B, heads*D, H, W]."""
    b_heads, head_dim, h, w = x.shape
    if b_heads % heads != 0:
        raise ValueError(f"batch-head dimension ({b_heads}) must be divisible by heads ({heads})")
    b = b_heads // heads
    return x.view(b, heads, head_dim, h, w).reshape(b, heads * head_dim, h, w)


def soft_orthogonalize(x: Tensor, strength: float = 0.05, eps: float = 1e-12) -> Tensor:
    """
    Apply one soft decorrelation step.

    This does not require num_axes <= feature_dim. It only nudges highly similar
    axes apart, then normalizes them again.

    Args:
        x: [heads, num_axes, feature_dim].
        strength: Decorrelation step size. 0 disables the update.
    """
    if strength <= 0:
        return F.normalize(x, dim=-1, eps=eps)

    axes = F.normalize(x, dim=-1, eps=eps)
    _, num_axes, _ = axes.shape
    eye = torch.eye(num_axes, dtype=axes.dtype, device=axes.device).unsqueeze(0)
    off_diag = torch.matmul(axes, axes.transpose(-2, -1)) - eye
    correction = torch.matmul(off_diag, axes) / max(num_axes - 1, 1)
    return F.normalize(axes - strength * correction, dim=-1, eps=eps)


def soft_orthogonal_loss(x: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Soft orthogonality loss for prototype axes.

    Args:
        x: [heads, num_axes, feature_dim] or [heads, C, K, feature_dim].

    Returns:
        Mean squared off-diagonal cosine similarity.
    """
    if x.dim() == 4:
        x = x.reshape(x.size(0), -1, x.size(-1))
    axes = F.normalize(x, dim=-1, eps=eps)
    num_axes = axes.size(1)
    gram = torch.matmul(axes, axes.transpose(-2, -1))
    eye = torch.eye(num_axes, dtype=gram.dtype, device=gram.device).unsqueeze(0)
    return ((gram - eye) ** 2).sum() / max(axes.size(0) * num_axes * (num_axes - 1), 1)


def resize_semantic_labels(labels: Tensor, size: Tuple[int, int]) -> Tensor:
    """Resize [B,H,W] or [B,1,H,W] semantic labels by nearest interpolation."""
    if labels.dim() == 4:
        if labels.size(1) != 1:
            raise ValueError(f"labels with 4 dims must have channel=1, got {tuple(labels.shape)}")
        labels = labels[:, 0]
    if labels.dim() != 3:
        raise ValueError(f"labels must be [B,H,W] or [B,1,H,W], got {tuple(labels.shape)}")

    labels = labels.float().unsqueeze(1)
    labels = F.interpolate(labels, size=size, mode="nearest")
    return labels[:, 0].long()


class PrototypeFeatureSpaceLoss(nn.Module):
    """
    CE + PPC + PPD supervision for SPACCv2 prototype feature space.

    CE classifies each pixel with class-level logits obtained by taking the max
    similarity over prototypes inside each semantic class.
    PPC classifies each pixel to its Sinkhorn-assigned class prototype.
    PPD pulls the pixel feature toward the assigned prototype by maximizing the
    assigned raw cosine similarity.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        ppc_weight: float = 0.01,
        ppd_weight: float = 0.001,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.ppc_weight = ppc_weight
        self.ppd_weight = ppd_weight
        self.ignore_index = ignore_index

    def forward(self, aux: Dict[str, Tensor], labels: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if "semantic_logits" not in aux or "prototype_logits" not in aux:
            raise KeyError("aux must contain 'semantic_logits' and 'prototype_logits'")

        semantic_logits = aux["semantic_logits"]
        prototype_logits = aux["prototype_logits"]
        prototype_similarity = aux.get("prototype_similarity", prototype_logits)
        prototype_target = aux.get("prototype_target")

        if semantic_logits.dim() != 5:
            raise ValueError(
                "semantic_logits must be [B, heads, num_classes, H, W], "
                f"got {tuple(semantic_logits.shape)}"
            )
        if prototype_logits.dim() != 5:
            raise ValueError(
                "prototype_logits must be [B, heads, num_classes*K, H, W], "
                f"got {tuple(prototype_logits.shape)}"
            )

        b, heads, num_classes, h, w = semantic_logits.shape
        num_axes = prototype_logits.size(2)
        fallback = semantic_logits.sum() * 0.0

        if labels is None:
            labels = aux.get("labels")
        if labels is None:
            loss_ce = fallback
        else:
            labels = resize_semantic_labels(labels, (h, w))
            valid = (labels != self.ignore_index) & (labels >= 0) & (labels < num_classes)
            labels_for_ce = labels.masked_fill(~valid, self.ignore_index)
            labels_for_ce = labels_for_ce.unsqueeze(1).expand(-1, heads, -1, -1)
            loss_ce = F.cross_entropy(
                semantic_logits.reshape(b * heads, num_classes, h, w),
                labels_for_ce.reshape(b * heads, h, w),
                ignore_index=self.ignore_index,
            )

        if prototype_target is None:
            loss_ppc = fallback
            loss_ppd = fallback
        else:
            prototype_target = prototype_target.long()
            loss_ppc = F.cross_entropy(
                prototype_logits.reshape(b * heads, num_axes, h, w),
                prototype_target.reshape(b * heads, h, w),
                ignore_index=self.ignore_index,
            )

            logits_flat = prototype_similarity.permute(0, 1, 3, 4, 2).reshape(-1, num_axes)
            target_flat = prototype_target.reshape(-1)
            valid = target_flat != self.ignore_index
            if torch.any(valid):
                assigned = torch.gather(logits_flat[valid], 1, target_flat[valid, None]).squeeze(1)
                loss_ppd = (1.0 - assigned).pow(2).mean()
            else:
                loss_ppd = fallback

        total = self.ce_weight * loss_ce + self.ppc_weight * loss_ppc + self.ppd_weight * loss_ppd
        return {
            "loss": total,
            "loss_ce": loss_ce,
            "loss_ppc": loss_ppc,
            "loss_ppd": loss_ppd,
        }


class SPACCv2(nn.Module):
    """
    Semantic Prototype Axes-guided Context Clustering v2.

    During training, if labels are provided, each class prototype is updated
    with EMA from assigned pixels. If return_aux=True, the module also returns
    the tensors needed for CE + PPC + PPD feature-space supervision.

    Input:
        x: [B, in_channel, H, W]
        labels: optional [B, H, W] or [B, 1, H, W], containing class ids.

    Output:
        out: [B, out_channel, H, W]
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: Optional[int] = None,
        num_classes: int = 4,
        prototypes_per_class: int = 5,
        heads: int = 5,
        head_dim: Optional[int] = None,
        momentum: float = 0.99,
        temperature: float = 0.07,
        update_temperature: float = 0.05,
        orthogonal_strength: float = 0.05,
        ignore_index: int = 255,
        update_prototypes: bool = True,
        sinkhorn_iterations: int = 3,
        loss_ce_weight: float = 1.0,
        loss_ppc_weight: float = 0.01,
        loss_ppd_weight: float = 0.001,
    ) -> None:
        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        if head_dim is None:
            if in_channel % heads != 0:
                raise ValueError("head_dim is None, so in_channel must be divisible by heads")
            head_dim = in_channel // heads
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if prototypes_per_class <= 0:
            raise ValueError("prototypes_per_class must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if update_temperature <= 0:
            raise ValueError("update_temperature must be positive")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.num_axes = num_classes * prototypes_per_class
        self.heads = heads
        self.head_dim = head_dim
        self.momentum = momentum
        self.temperature = temperature
        self.update_temperature = update_temperature
        self.orthogonal_strength = orthogonal_strength
        self.ignore_index = ignore_index
        self.update_enabled = update_prototypes
        self.sinkhorn_iterations = sinkhorn_iterations
        self.loss_ce_weight = loss_ce_weight
        self.loss_ppc_weight = loss_ppc_weight
        self.loss_ppd_weight = loss_ppd_weight

        self.q = nn.Conv2d(in_channel, heads * head_dim, kernel_size=1)
        self.v = nn.Conv2d(in_channel, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv2d(heads * head_dim, out_channel, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

        prototypes = torch.randn(heads, num_classes, prototypes_per_class, head_dim)
        prototypes = F.normalize(prototypes, dim=-1)
        self.prototypes = nn.Parameter(prototypes, requires_grad=False)

    def orthogonality_loss(self) -> Tensor:
        return soft_orthogonal_loss(self.prototypes)

    def _prepare_labels(self, labels: Tensor, size: Tuple[int, int]) -> Tensor:
        return resize_semantic_labels(labels, size)

    @torch.no_grad()
    def _build_prototype_targets(self, prototype_similarity: Tensor, labels: Tensor) -> Tensor:
        """
        Build ProtoSeg-style prototype targets for PPC/PPD.

        Args:
            prototype_similarity: [B*heads, num_classes*K, H*W] raw cosine similarity.
            labels: [B,H,W] semantic labels resized to feature resolution.

        Returns:
            [B, heads, H, W] target indices in [0, num_classes*K), or ignore_index.
        """
        b, h, w = labels.shape
        n = h * w
        scores = prototype_similarity.reshape(
            b,
            self.heads,
            self.num_classes,
            self.prototypes_per_class,
            n,
        )
        flat_labels = labels.reshape(b, n)
        targets = torch.full(
            (b, self.heads, n),
            fill_value=self.ignore_index,
            dtype=torch.long,
            device=labels.device,
        )

        valid_label = (flat_labels != self.ignore_index) & (flat_labels >= 0) & (flat_labels < self.num_classes)
        for batch_idx in range(b):
            for head_idx in range(self.heads):
                for class_idx in range(self.num_classes):
                    class_mask = valid_label[batch_idx] & (flat_labels[batch_idx] == class_idx)
                    if not torch.any(class_mask):
                        continue

                    class_scores = scores[batch_idx, head_idx, class_idx, :, class_mask].transpose(0, 1)
                    _, indices = distributed_sinkhorn(
                        class_scores,
                        sinkhorn_iterations=self.sinkhorn_iterations,
                        epsilon=self.update_temperature,
                    )
                    targets[batch_idx, head_idx, class_mask] = (
                        class_idx * self.prototypes_per_class + indices.long()
                    )

        return targets.reshape(b, self.heads, h, w)

    @torch.no_grad()
    def _ema_update_prototypes(self, feat: Tensor, labels: Tensor, pred_classes: Tensor) -> None:
        labels = self._prepare_labels(labels, feat.shape[-2:])
        b, h, w = labels.shape
        if pred_classes.shape != (b, self.heads, h, w):
            raise ValueError(
                "pred_classes must have shape "
                f"{(b, self.heads, h, w)}, got {tuple(pred_classes.shape)}"
            )

        feat = split_heads_2d(feat, self.heads)
        feat = feat.reshape(b, self.heads, self.head_dim, h * w).permute(1, 0, 3, 2)
        feat = F.normalize(feat, dim=-1)
        labels = labels.reshape(b, h * w)
        pred_classes = pred_classes.long().reshape(b, self.heads, h * w).permute(1, 0, 2)

        valid_label = (labels != self.ignore_index) & (labels >= 0) & (labels < self.num_classes)
        prototypes = F.normalize(self.prototypes, dim=-1).detach().clone()

        for head_idx in range(self.heads):
            head_feat = feat[head_idx]
            head_pred = pred_classes[head_idx]
            head_prototypes = prototypes[head_idx]

            for class_idx in range(self.num_classes):
                class_mask = valid_label & (labels == class_idx)
                class_pixels = head_feat[class_mask]
                if class_pixels.numel() == 0:
                    continue

                old_proto = head_prototypes[class_idx]
                assign_logits = torch.matmul(class_pixels, old_proto.transpose(0, 1))
                assign, _ = distributed_sinkhorn(
                    assign_logits,
                    sinkhorn_iterations=self.sinkhorn_iterations,
                    epsilon=self.update_temperature,
                )
                correct_pixels = (head_pred[class_mask] == class_idx).float()
                assign = assign * correct_pixels.unsqueeze(1)
                mass = assign.sum(dim=0)
                valid_proto = mass > 1e-6
                if not torch.any(valid_proto):
                    continue

                new_proto = torch.matmul(assign.transpose(0, 1), class_pixels)
                new_proto = F.normalize(new_proto, dim=-1)

                updated = (
                    self.momentum * old_proto[valid_proto]
                    + (1.0 - self.momentum) * new_proto[valid_proto]
                )
                head_prototypes[class_idx, valid_proto] = updated

        self.prototypes.copy_(F.normalize(prototypes, dim=-1))

    def forward(
        self,
        x: Tensor,
        labels: Optional[Tensor] = None,
        return_assignment: bool = False,
        return_shape_info: bool = False,
        return_aux: bool = False,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tensor],
        Tuple[Tensor, Dict[str, Tensor]],
        Tuple[Tensor, Tensor, Dict[str, Tuple[int, ...]]],
    ]:
        if x.dim() != 4:
            raise ValueError(f"SPACC expects a 4D tensor [B,C,H,W], got shape {tuple(x.shape)}")

        b0, _, h0, w0 = x.shape
        query_full = self.q(x)
        value = split_heads_2d(self.v(x), self.heads)
        query = split_heads_2d(query_full, self.heads)
        b_heads, c, h, w = query.shape

        pixels = query.reshape(b_heads, c, -1).permute(0, 2, 1)
        value_pixels = value.flatten(2).transpose(1, 2)

        prototypes = F.normalize(self.prototypes.detach(), dim=-1)
        proto_flat = prototypes.reshape(self.heads, self.num_axes, self.head_dim)
        proto_flat = proto_flat.unsqueeze(0).expand(b0, -1, -1, -1).reshape(b_heads, self.num_axes, self.head_dim)

        prototype_similarity = pairwise_cos_sim(proto_flat, pixels)
        sim_logits = self.sim_beta + self.sim_alpha * prototype_similarity
        assignment = F.softmax(sim_logits / self.temperature, dim=1)

        axis_idx = assignment.argmax(dim=1)
        assignment_map = axis_idx.reshape(b0, self.heads, h0, w0)
        class_assignment_map = (axis_idx // self.prototypes_per_class).reshape(b0, self.heads, h0, w0)

        clustered = torch.matmul(assignment, value_pixels)
        clustered = clustered / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        dispatched = torch.matmul(assignment.transpose(1, 2), clustered)
        out = dispatched.transpose(1, 2).reshape(b_heads, c, h, w)
        out = merge_heads_2d(out, self.heads)
        out = self.proj(out)

        if self.training and self.update_enabled and labels is not None:
            self._ema_update_prototypes(query_full.detach(), labels.detach(), class_assignment_map.detach())

        prototype_logits = sim_logits.reshape(b0, self.heads, self.num_axes, h, w)
        prototype_similarity_map = prototype_similarity.reshape(b0, self.heads, self.num_axes, h, w)
        semantic_logits = prototype_logits.reshape(
            b0,
            self.heads,
            self.num_classes,
            self.prototypes_per_class,
            h,
            w,
        ).amax(dim=3)
        prototype_target = None
        prepared_labels = None
        if labels is not None:
            prepared_labels = self._prepare_labels(labels, (h, w))
            prototype_target = self._build_prototype_targets(prototype_similarity, prepared_labels)

        aux = {
            "soft_orthogonal_loss": self.orthogonality_loss(),
            "assignment_map": assignment_map,
            "class_assignment_map": class_assignment_map,
            "semantic_logits": semantic_logits,
            "prototype_logits": prototype_logits,
            "prototype_similarity": prototype_similarity_map,
            "prototype_target": prototype_target,
            "labels": prepared_labels,
        }

        if not return_assignment and not return_shape_info and not return_aux:
            return out

        shape_info = {
            "input": tuple(x.shape),
            "q_or_v_after_conv": (b0, self.heads * self.head_dim, h0, w0),
            "after_split_heads": tuple(query.shape),
            "prototypes": tuple(self.prototypes.shape),
            "prototype_axes": tuple(proto_flat.shape),
            "pixels": tuple(pixels.shape),
            "similarity": tuple(sim_logits.shape),
            "semantic_logits": tuple(semantic_logits.shape),
            "prototype_logits": tuple(prototype_logits.shape),
            "assignment": tuple(assignment.shape),
            "clustered": tuple(clustered.shape),
            "assignment_map": tuple(assignment_map.shape),
            "class_assignment_map": tuple(class_assignment_map.shape),
            "output": tuple(out.shape),
        }

        outputs = [out]
        if return_assignment:
            outputs.append(class_assignment_map)
        if return_shape_info:
            outputs.append(shape_info)
        if return_aux:
            outputs.append(aux)
        return tuple(outputs)

    def feature_space_loss(self, aux: Dict[str, Tensor], labels: Optional[Tensor] = None) -> Dict[str, Tensor]:
        criterion = PrototypeFeatureSpaceLoss(
            ce_weight=self.loss_ce_weight,
            ppc_weight=self.loss_ppc_weight,
            ppd_weight=self.loss_ppd_weight,
            ignore_index=self.ignore_index,
        )
        return criterion(aux, labels)


SPACC = SPACCv2
SemanticPrototypeContextClusterModule = SPACCv2


def _count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _run_spacc_shape_case(name: str, shape: Tuple[int, int, int, int], module_kwargs: Dict[str, int]) -> None:
    print("=" * 80)
    print(name)
    print("module kwargs:", module_kwargs)

    x = torch.randn(*shape)
    labels = torch.randint(
        low=0,
        high=module_kwargs.get("num_classes", 4),
        size=(shape[0], shape[2], shape[3]),
    )
    module = SPACC(**module_kwargs).train()

    proto_before = module.prototypes.clone()
    y, class_assignment_map, shape_info, aux = module(
        x,
        labels=labels,
        return_assignment=True,
        return_shape_info=True,
        return_aux=True,
    )
    proto_delta = (module.prototypes - proto_before).abs().sum().item()

    print("trainable params:", _count_trainable_params(module))
    print("semantic prototypes shape:", tuple(module.prototypes.shape))
    print("soft orthogonal loss:", float(aux["soft_orthogonal_loss"]))
    print("prototype update delta:", proto_delta)
    print("unique assigned classes in first sample/head:", torch.unique(class_assignment_map[0, 0]).tolist())
    print("\nshape flow:")
    for key, value in shape_info.items():
        print(f"  {key:22s}: {value}")

    expected_output_shape = (shape[0], module_kwargs["out_channel"], shape[2], shape[3])
    expected_assignment_shape = (shape[0], module_kwargs["heads"], shape[2], shape[3])
    assert tuple(y.shape) == expected_output_shape
    assert tuple(class_assignment_map.shape) == expected_assignment_shape
    assert tuple(module.prototypes.shape) == (
        module_kwargs["heads"],
        module_kwargs.get("num_classes", 4),
        module_kwargs.get("prototypes_per_class", 5),
        module_kwargs["head_dim"],
    )
    assert proto_delta > 0.0

    module.eval()
    proto_before_eval = module.prototypes.clone()
    with torch.no_grad():
        y_eval, eval_assignment = module(x, return_assignment=True)
    proto_eval_delta = (module.prototypes - proto_before_eval).abs().sum().item()
    assert tuple(y_eval.shape) == expected_output_shape
    assert tuple(eval_assignment.shape) == expected_assignment_shape
    assert proto_eval_delta == 0.0
    print("eval prototype update delta:", proto_eval_delta)


def demo_spacc_shapes() -> None:
    torch.manual_seed(0)

    _run_spacc_shape_case(
        "SPACC Landsat-SCD style semantic prototypes",
        shape=(2, 64, 32, 32),
        module_kwargs=dict(
            in_channel=64,
            out_channel=64,
            num_classes=4,
            prototypes_per_class=5,
            heads=5,
            head_dim=16,
        ),
    )

    _run_spacc_shape_case(
        "SPACC deeper feature stage",
        shape=(1, 256, 16, 16),
        module_kwargs=dict(
            in_channel=256,
            out_channel=256,
            num_classes=4,
            prototypes_per_class=5,
            heads=5,
            head_dim=64,
        ),
    )

    print("=" * 80)
    print("All SPACC shape/update tests passed.")


if __name__ == "__main__":
    demo_spacc_shapes()

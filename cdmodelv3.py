import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        key_in_channels,
        query_in_channels,
        transform_channels,
        out_channels,
        key_query_num_convs,
        value_out_num_convs,
    ):
        super(SelfAttentionBlock, self).__init__()
        self.key_project = self._build_project(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.query_project = self._build_project(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.value_project = self._build_project(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=value_out_num_convs,
        )
        self.out_project = self._build_project(
            in_channels=transform_channels,
            out_channels=out_channels,
            num_convs=value_out_num_convs,
        )
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])

        return self.out_project(context)

    def _build_project(self, in_channels, out_channels, num_convs):
        layers = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for _ in range(num_convs - 1):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        return nn.Sequential(*layers)


class ImprovedTemporalFeatureFusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.diff_proj = dsconv_3x3(in_channel * 5, in_channel)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, max(in_channel // 4, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channel // 4, 1), in_channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.out_conv = dsconv_3x3(in_channel * 4, out_channel)

    def forward(self, xA, xB):
        diff_abs = torch.abs(xA - xB)
        diff_ab = xA - xB
        diff_ba = xB - xA

        diff_feat = self.diff_proj(
            torch.cat([xA, xB, diff_abs, diff_ab, diff_ba], dim=1)
        )

        s_gate = self.spatial_gate(diff_feat)
        c_gate = self.channel_gate(diff_feat)

        xA_w = xA * s_gate * c_gate
        xB_w = xB * s_gate * c_gate

        out = self.out_conv(
            torch.cat([xA_w, xB_w, diff_abs, diff_feat], dim=1)
        )

        return out


class ImprovedSpatialFeatureFusion(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.conv_small = conv_1x1(in_channel, in_channel)
        self.conv_big = conv_1x1(in_channel, in_channel)

        self.value_fuse = conv_1x1(in_channel * 2, in_channel)

        self.attention = SelfAttentionBlock(
            key_in_channels=in_channel,
            query_in_channels=in_channel,
            transform_channels=max(in_channel // 2, 1),
            out_channels=in_channel,
            key_query_num_convs=2,
            value_out_num_convs=1,
        )

        self.out_conv = conv_3x3(in_channel * 2, in_channel)

    def forward(self, x_small, x_big):
        img_size = x_big.size(2), x_big.size(3)

        x_small = F.interpolate(
            x_small,
            size=img_size,
            mode="bilinear",
            align_corners=False
        )

        small = self.conv_small(x_small)
        big = self.conv_big(x_big)

        qk = small + big
        value = self.value_fuse(torch.cat([small, big], dim=1))

        new_x = self.attention(qk, qk, value)

        return self.out_conv(torch.cat([new_x, big], dim=1))


class CDModel(nn.Module):
    """Fuse TFF and SFF into one change-detection feature fusion module."""

    def __init__(self, in_channel, out_channel=None, context_channel=None):
        super(CDModel, self).__init__()
        out_channel = out_channel or in_channel
        context_channel = context_channel or out_channel

        self.temporal_fusion = ImprovedTemporalFeatureFusion(in_channel, out_channel)
        self.context_project = (
            nn.Identity()
            if context_channel == out_channel
            else conv_1x1(context_channel, out_channel)
        )
        self.spatial_fusion = ImprovedSpatialFeatureFusion(out_channel)

    def forward(self, xA, xB, x_context=None):
        if xA.shape != xB.shape:
            raise ValueError(f"xA and xB must have the same shape, got {xA.shape} and {xB.shape}.")

        temporal_feat = self.temporal_fusion(xA, xB)
        if x_context is None:
            x_context = temporal_feat
        else:
            x_context = self.context_project(x_context)

        return self.spatial_fusion(x_context, temporal_feat)


class PriorGuidedChangeFusion(nn.Module):
    """Learnable prior-guided fusion for semantic-temporal change features."""

    def __init__(self, in_channel, out_channel=None, gate_hidden_channel=None):
        super(PriorGuidedChangeFusion, self).__init__()
        out_channel = out_channel or in_channel
        gate_hidden_channel = gate_hidden_channel or max(in_channel // 2, 1)

        self.gate = nn.Sequential(
            conv_1x1(in_channel * 2, gate_hidden_channel),
            nn.Conv2d(gate_hidden_channel, in_channel, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = conv_3x3(in_channel * 3, out_channel)

    def forward(self, prior_feat, semantic_temporal_feat, return_gate=False):
        if prior_feat.shape != semantic_temporal_feat.shape:
            raise ValueError(
                "prior_feat and semantic_temporal_feat must have the same shape, "
                f"got {prior_feat.shape} and {semantic_temporal_feat.shape}."
            )

        gate = self.gate(torch.cat([prior_feat, semantic_temporal_feat], dim=1))
        fused = self.fuse(
            torch.cat([gate * semantic_temporal_feat, prior_feat, semantic_temporal_feat], dim=1)
        )

        if return_gate:
            return fused, gate
        return fused


cdmodel = CDModel
TFFSFFFusion = CDModel

__all__ = [
    "CDModel",
    "PriorGuidedChangeFusion",
    "TFFSFFFusion",
    "cdmodel",
    "ImprovedTemporalFeatureFusion",
    "ImprovedSpatialFeatureFusion",
    "SelfAttentionBlock",
]

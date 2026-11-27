import torch
import torch.nn as nn


class CMConv(nn.Module):
    """Channel masked convolution used by SGAM's residual conv block."""

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=3,
        groups=1,
        dilation_set=4,
        bias=False,
    ):
        super(CMConv, self).__init__()
        if in_ch % (groups * dilation_set) != 0:
            raise ValueError("in_ch must be divisible by groups * dilation_set")
        if out_ch % (groups * dilation_set) != 0:
            raise ValueError("out_ch must be divisible by groups * dilation_set")
        if in_ch % (2 * groups) != 0:
            raise ValueError("in_ch must be divisible by 2 * groups")

        self.prim = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding=dilation,
            dilation=dilation,
            groups=groups * dilation_set,
            bias=bias,
        )
        self.prim_shift = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding=2 * dilation,
            dilation=2 * dilation,
            groups=groups * dilation_set,
            bias=bias,
        )
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )

        mask = torch.zeros_like(self.conv.weight, dtype=torch.bool)
        in_channels_per_set = in_ch // (groups * dilation_set)
        out_channels_per_set = out_ch // (groups * dilation_set)

        for i in range(dilation_set):
            for j in range(groups):
                out_start = (i + j * groups) * out_channels_per_set
                out_end = (i + j * groups + 1) * out_channels_per_set
                in_start = i * in_channels_per_set
                in_end = (i + 1) * in_channels_per_set
                mask[out_start:out_end, in_start:in_end, :, :] = True

                shifted_i = (i + dilation_set // 2) % dilation_set
                shifted_out_start = (shifted_i + j * groups) * out_channels_per_set
                shifted_out_end = (
                    shifted_i + j * groups + 1
                ) * out_channels_per_set
                mask[shifted_out_start:shifted_out_end, in_start:in_end, :, :] = True

        self.register_buffer("mask", mask, persistent=False)
        with torch.no_grad():
            self.conv.weight.masked_fill_(self.mask, 0)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.conv.weight.register_hook(backward_hook)
        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(
            tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split),
            dim=1,
        )
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift


class SGAM_Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SGAM_Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(out_ch)
        self.ReLU = nn.ReLU(inplace=False)
        self.conv2 = CMConv(out_ch, out_ch, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        return self.ReLU(x + self.BN2(self.conv2(self.ReLU(self.BN1(x)))))


class GaoSi_core(nn.Module):
    def __init__(self, in_ch):
        super(GaoSi_core, self).__init__()

    def forward(self, mutual_feature, feature):
        _, _, h, w = feature.size()
        mean = mutual_feature.mean(dim=[2, 3], keepdim=True)
        square = (feature - mean).pow(2)
        sigma = square.sum(dim=[2, 3], keepdim=True) / (h * w)
        att_score = square / (2 * sigma + 1e-8) + 0.5
        att_weight = torch.sigmoid(att_score)
        return att_weight * feature


class SGAM(nn.Module):
    """Spatial Gaussian Attention Module for dual-temporal feature fusion."""

    def __init__(self, in_ch, out_ch):
        super(SGAM, self).__init__()
        if out_ch % 2 != 0:
            raise ValueError("out_ch must be even")

        native_ch = out_ch // 2
        self.SGAM_conv = nn.Conv2d(in_ch, native_ch, kernel_size=1)
        self.BN1 = nn.BatchNorm2d(native_ch)
        self.ReLU = nn.ReLU(inplace=True)
        self.GaoSi = GaoSi_core(native_ch)
        self.conv_finally = SGAM_Conv_Block(out_ch, out_ch)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, feature1, feature2):
        feat1 = self.SGAM_conv(feature1)
        feat2 = self.SGAM_conv(feature2)
        feat1_wave = self.ReLU(self.BN1(feat1))
        feat2_wave = self.ReLU(self.BN1(feat2))

        mutual_feature = (feat1_wave + feat2_wave) * 0.5
        feat1_hat = self.GaoSi(mutual_feature, feat1)
        feat2_hat = self.GaoSi(mutual_feature, feat2)
        result = torch.cat(
            [feat1_hat * self.beta + feat1, feat2_hat * self.beta + feat2],
            dim=1,
        )
        return self.conv_finally(result)


SGAMConvBlock = SGAM_Conv_Block
GaoSiCore = GaoSi_core

__all__ = [
    "CMConv",
    "SGAMConvBlock",
    "SGAM_Conv_Block",
    "GaoSiCore",
    "GaoSi_core",
    "SGAM",
]

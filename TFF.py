import torch
from torch import nn


def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channel,
        ),
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


class SaELayer(nn.Module):
    def __init__(self, in_channel, reduction=32):
        super(SaELayer, self).__init__()
        assert in_channel >= reduction and in_channel % reduction == 0, (
            "invalid in_channel in SaElayer"
        )
        self.reduction = reduction
        self.cardinality = 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(
                in_channel // self.reduction * self.cardinality,
                in_channel,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y_concate = torch.cat([y1, y2, y3, y4], dim=1)
        y_ex_dim = self.fc(y_concate).view(b, c, 1, 1)

        return y_ex_dim.expand_as(x)


class TFF(nn.Module):
    """Basic temporal feature fusion used in stnet.py and DDLNet.py."""

    def __init__(self, in_channel, out_channel):
        super(TFF, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB):
        x_diff = xA - xB

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA
        xB = B_weight * xB

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x


class TFFWithSaE(nn.Module):
    """TFF variant with SaELayer used in cdmask.py and MaskFormerModel_sea_ourDH.py."""

    def __init__(self, in_channel, out_channel):
        super(TFFWithSaE, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.senetv2 = SaELayer(in_channel)

    def forward(self, xA, xB):
        x_diff = xA - xB
        x_weight = self.senetv2(x_diff)

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA * x_weight
        xB = B_weight * xB * x_weight

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x


BasicTFF = TFF
EnhancedTFF = TFFWithSaE
TemporalFeatureFusion = TFF

__all__ = [
    "dsconv_3x3",
    "SaELayer",
    "TFF",
    "TFFWithSaE",
    "BasicTFF",
    "EnhancedTFF",
    "TemporalFeatureFusion",
]

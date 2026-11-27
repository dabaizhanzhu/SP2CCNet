import timm
import os
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torchvision import models
from SPACCv2 import SPACCv2
from cdmodelv3 import CDModel, PriorGuidedChangeFusion


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class Residual(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super(Residual, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class BGM(nn.Module):
    def __init__(self, in_dim):
        super(BGM, self).__init__()
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, boundary):
        m_batchsize, C, height, width = x.size()
        q = self.query_conv1(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k = self.key_conv1(boundary).view(m_batchsize, -1, width * height)
        v = self.value_conv1(x).view(m_batchsize, -1, width * height)
        energy1 = torch.bmm(q, k)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)
        out1 = x + out1

        return out1



class CotSR(nn.Module):
    def __init__(self, in_dim):
        super(CotSR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        ''' inputs :
                x1 : input feature maps( B X C X H X W)
                x2 : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x1.size()

        q1 = self.query_conv1(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width * height)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width * height)

        q2 = self.query_conv2(x2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width * height)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width * height)

        energy1 = torch.bmm(q1, k2)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v2, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)

        out1 = x1 + self.gamma1 * out1
        out2 = x2 + self.gamma2 * out2

        return out1, out2


class multi_scale(nn.Module):
    '''
    Three-layer CNN for encoding map info.
    '''
    def __init__(
        self,
        input_channels,
        out_channels,
    ):
        super(multi_scale, self).__init__()
        self.out_channels = out_channels
        self.input_channels= input_channels
        self.layer1 = BasicConv2d(input_channels,out_channels,1)
        self.layer2 = BasicConv2d(out_channels,out_channels,3, 1, 1, 1)
        self.layer3 = BasicConv2d(out_channels, out_channels, 5, 1, 2, 1)

    def forward(self, edge):
        x = self.layer1(edge)
        x1 = edge + x
        x2 = self.layer2(x1)
        x3 = x2 + x1
        x4 = self.layer3(x3)
        x_total = x4 + x3

        return x_total


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


class LaplaceConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(LaplaceConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

        laplace_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
        laplace_kernel = laplace_kernel.unsqueeze(0).unsqueeze(0)
        laplace_kernel = laplace_kernel.repeat((out_channels, in_channels, 1, 1))
        self.conv.weight = nn.Parameter(laplace_kernel)
        self.conv.bias.data.fill_(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.relu(self.bn(x1))

        return x1





import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img = img[:, :, ::-1]
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))


def _load_pretrained_weights(backbone, pretrained_path):
    if not pretrained_path or not os.path.exists(pretrained_path):
        warnings.warn(f'PVT pretrained weights not found: {pretrained_path}. Using random initialization.')
        return backbone

    save_model = torch.load(pretrained_path, map_location='cpu')
    if isinstance(save_model, dict) and 'state_dict' in save_model:
        save_model = save_model['state_dict']

    model_dict = backbone.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    backbone.load_state_dict(model_dict)
    return backbone



class CD_branch(nn.Module):
    def __init__(self, channels, pretrained_path):
        super(CD_branch, self).__init__()
        self.backbone_cd = pvt_v2_b2()
        self.backbone_cd = _load_pretrained_weights(self.backbone_cd, pretrained_path)
        self.Translayer2_11 = BasicConv2d(768, channels, 3, padding=1)
        self.Translayer3_11 = BasicConv2d(480, channels, 3, padding=1)
        self.Translayer4_11 = BasicConv2d(192, channels, 3, padding=1)
        self.Translayer5_11 = BasicConv2d(192, channels, 3, padding=1)
        self.Translayer0_11 = BasicConv2d(6, 3, 1)

        self.up11 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up21 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up31 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.LAM31 = LAM(512, 256)
        self.LAM21 = LAM(320, 160)
        self.LAM11 = LAM(128, 64)

        self.C31 = CBAM(512)
        self.C21 = CBAM(320)
        self.C11 = CBAM(128)
        self.C01 = CBAM(64)

    def forward(self, t1, t2, return_level1=False):
        x_s = torch.cat((t1, t2), 1)
        x_s = self.Translayer0_11(x_s)
        pvt = self.backbone_cd(x_s)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        cim_feature = self.C01(x1)
        x2_1 = self.LAM11(x2)
        x2_2 = self.C11(x2)
        x2_3 = torch.cat((x2_1, x2_2), 1)
        x2_3 = self.Translayer4_11(x2_3)
        x2_4 = self.up11(x2_3)

        x3_1 = self.LAM21(x3)
        x3_2 = self.C21(x3)
        x3_3 = torch.cat((x3_1, x3_2), 1)
        x3_3 = self.Translayer3_11(x3_3)
        x3_4 = self.up21(x3_3)

        x4_1 = self.LAM31(x4)
        x4_2 = self.C31(x4)
        x4_3 = torch.cat((x4_1, x4_2), 1)
        x4_3 = self.Translayer2_11(x4_3)
        x4_4 = self.up31(x4_3)

        x_total = torch.cat((x2_4, x3_4, x4_4), 1)
        x_high = self.Translayer5_11(x_total)
        x_low = torch.nn.functional.softmax(cim_feature, dim=1)
        x_fuse = x_high + (x_high * x_low)
        if return_level1:
            return x_fuse, cim_feature
        return x_fuse


class LAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LAM, self).__init__()

        self.conv1 = BasicConv2d(in_channels // 2, out_channels, 1)

        self.deconv1 = nn.Conv2d(
            in_channels, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels, in_channels // 8, (1, 9), padding=(0, 4)
        )

    def forward(self, x):
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv1(x)

        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class CPM(nn.Module):
    def __init__(self, in_dim):
        super(CPM, self).__init__()
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, boundary):
        m_batchsize, C, height, width = x.size()
        q = self.query_conv1(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k = self.key_conv1(boundary).view(m_batchsize, -1, width * height)
        v = self.value_conv1(x).view(m_batchsize, -1, width * height)
        energy1 = torch.bmm(q, k)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)
        out1 = x + out1

        return out1

class CPGNet(nn.Module):
    def __init__(self, channel=32, num_classes=7, pretrained_path=r'D:\SCD\pretrained\pvt_v2_b2.pth', drop_rate=0.4):
        super(CPGNet, self).__init__()
        self.drop = nn.Dropout2d(drop_rate)
        self.backbone = pvt_v2_b2()
        self.backbone = _load_pretrained_weights(self.backbone, pretrained_path)

        self.channel = channel
        self.num_classes = num_classes

        self.Translayer2_1 = BasicConv2d(768, channel, 3, padding=1)
        self.Translayer3_1 = BasicConv2d(480, channel, 3, padding=1)
        self.Translayer4_1 = BasicConv2d(192, channel, 3, padding=1)

        self.Translayer5_1 = BasicConv2d(192, channel, 3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.out_feature = nn.Conv2d(64, num_classes, 1)
        self.out_feature2 = nn.Conv2d(64, num_classes, 1)
        self.out_feature1 = nn.Conv2d(64, 1, 1)

        self.LAM3 = LAM(512, 256)
        self.LAM2 = LAM(320, 160)
        self.LAM1 = LAM(128, 64)

        self.C3 = SPACCv2(512, num_classes=num_classes, heads=8)
        self.C2 = SPACCv2(320, num_classes=num_classes, heads=8)
        self.C1 = SPACCv2(128, num_classes=num_classes, heads=8)
        self.C0 = CBAM(64)

        self.cdmodel = CDModel(64, 64, context_channel=64)
        self.change_fusion = PriorGuidedChangeFusion(64, 64)

        self.cd_branch = CD_branch(channel, pretrained_path)

    def base_forward(self, x, labels=None, return_spacc_loss=False):
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        spacc_losses = []
        cim_feature = self.C0(x1)
        x2_1 = self.LAM1(x2)
        if return_spacc_loss:
            x2_2, aux2 = self.C1(x2, labels=labels, return_aux=True)
            spacc_losses.append(self.C1.feature_space_loss(aux2, labels))
        else:
            x2_2 = self.C1(x2, labels=labels)
        x2_3 = torch.cat((x2_1, x2_2), 1)
        x2_3 = self.Translayer4_1(x2_3)
        x2_4 = self.up1(x2_3)

        x3_1 = self.LAM2(x3)
        if return_spacc_loss:
            x3_2, aux3 = self.C2(x3, labels=labels, return_aux=True)
            spacc_losses.append(self.C2.feature_space_loss(aux3, labels))
        else:
            x3_2 = self.C2(x3, labels=labels)
        x3_3 = torch.cat((x3_1, x3_2), 1)
        x3_3 = self.Translayer3_1(x3_3)
        x3_4 = self.up2(x3_3)

        x4_1 = self.LAM3(x4)
        if return_spacc_loss:
            x4_2, aux4 = self.C3(x4, labels=labels, return_aux=True)
            spacc_losses.append(self.C3.feature_space_loss(aux4, labels))
        else:
            x4_2 = self.C3(x4, labels=labels)
        x4_3 = torch.cat((x4_1, x4_2), 1)
        x4_3 = self.Translayer2_1(x4_3)
        x4_4 = self.up3(x4_3)

        x_total = torch.cat((x2_4, x3_4, x4_4), 1)
        x_high = self.Translayer5_1(x_total)
        x_low = torch.nn.functional.softmax(cim_feature, dim=1)
        x_fuse = x_high + (x_high * x_low)
        if return_spacc_loss:
            return x_fuse, spacc_losses
        return x_fuse

    def _merge_spacc_losses(self, spacc_losses):
        return {
            key: torch.stack([loss_item[key] for loss_item in spacc_losses]).mean()
            for key in ("loss", "loss_ce", "loss_ppc", "loss_ppd")
        }

    def forward(self, t1, t2, labels_A=None, labels_B=None, return_spacc_loss=False):
        if return_spacc_loss:
            t1_out, t1_spacc_losses = self.base_forward(t1, labels=labels_A, return_spacc_loss=True)
            t2_out, t2_spacc_losses = self.base_forward(t2, labels=labels_B, return_spacc_loss=True)
            spacc_loss = self._merge_spacc_losses(t1_spacc_losses + t2_spacc_losses)
        else:
            t1_out = self.base_forward(t1, labels=labels_A)
            t2_out = self.base_forward(t2, labels=labels_B)
            spacc_loss = None
        cd_out, cd_level1 = self.cd_branch(t1, t2, return_level1=True)

        changes = self.cdmodel(t1_out, t2_out, x_context=cd_level1)

        Change_cd = self.change_fusion(cd_out, changes)

        out_cs = self.out_feature1(Change_cd)

        output1 = self.out_feature(t1_out)
        output2 = self.out_feature2(t2_out)

        prediction1_1 = F.interpolate(out_cs, scale_factor=4, mode='bilinear')
        prediction1_2 = F.interpolate(output1, scale_factor=4, mode='bilinear')
        prediction1_3 = F.interpolate(output2, scale_factor=4, mode='bilinear')

        if return_spacc_loss:
            return prediction1_1, prediction1_2, prediction1_3, spacc_loss
        return prediction1_1, prediction1_2, prediction1_3

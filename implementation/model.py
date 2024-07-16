import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Leaky_ReLU6(nn.Module):
    def __init__(self, a_value=0.001):
        super(Leaky_ReLU6, self).__init__()
        self.a = a_value
    def forward(self, x):
      quantized_six = torch.tensor(6.0).to(x.device)
      out = torch.min(quantized_six, torch.max(self.a*x, x))
      return out

class leaky_hswish(nn.Module):
    def __init__(self, a_value=0.001):
        super(leaky_hswish, self).__init__()
        self.leakyrelu6 = Leaky_ReLU6(a_value)
    def forward(self, x):
        out = x * self.leakyrelu6(x + 3) / 6
        return out

class leaky_hsigmoid(nn.Module):
    def __init__(self, a_value=0.001):
        super(leaky_hsigmoid, self).__init__()
        self.leakyrelu6 = Leaky_ReLU6(a_value)
    def forward(self, x):
        out = self.leakyrelu6(x + 3) / 6
        return out

class SeModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SeModule, self).__init__()
        reduced_channels = max(in_channels // reduction, 8)
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            leaky_hsigmoid()
        )

    def forward(self, x):
        return x * self.se_block(x)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            leaky_hsigmoid()
        )
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale

class CBAMModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMModule, self).__init__()
        self.channel_gate = SeModule(in_channels, reduction)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        F_1 = self.channel_gate(x)
        F_2 = self.spatial_gate(F_1)
        return F_2


class BSConvU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
          super(BSConvU, self).__init__()
          self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
          self.bn1 = nn.BatchNorm2d(out_channels)
          self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=out_channels, bias=False)
          self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pointwise_conv(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        return x


class CustomBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, expand_channels, out_channels, activation, se, stride, leakyrelu):
        super(CustomBlock, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_channels)
        self.act1 = activation(inplace=True) if not leakyrelu else activation(a_value=0.001)

        self.bsconv = BSConvU(expand_channels, expand_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

        self.se = SeModule(expand_channels) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = activation(inplace=True) if not leakyrelu else activation(a_value=0.001)

        self.cbam = CBAMModule(out_channels)

        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif stride == 2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_channels)
            ) if in_channels != out_channels else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=in_channels, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = x

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bsconv(x)
        x = self.se(x)
        x = self.bn3(self.conv3(x))

        x = self.cbam(x)

        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        return self.act3(x + shortcut)



class ModifiedMobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1, activation=leaky_hswish):
        super(ModifiedMobileNetV3_Small, self).__init__()
        self.initial_conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(16)
        self.initial_act = activation(a_value=0.001)

        self.bneck = nn.Sequential(
            CustomBlock(3, 16, 16, 16, nn.ReLU, True, 2, False),
            CustomBlock(3, 16, 72, 24, nn.ReLU, False, 2, False),
            CustomBlock(3, 24, 88, 24, nn.ReLU, False, 1, False),
            CustomBlock(5, 24, 96, 40, activation, True, 2, True),
            CustomBlock(5, 40, 240, 40, activation, True, 1, True),
            CustomBlock(5, 40, 240, 40, activation, True, 1, True),
            CustomBlock(5, 40, 120, 48, activation, True, 1, True),
            CustomBlock(5, 48, 144, 48, activation, True, 1, True),
            CustomBlock(5, 48, 288, 96, activation, True, 2, True),
            CustomBlock(5, 96, 576, 96, activation, True, 1, True),
            CustomBlock(5, 96, 576, 96, activation, True, 1, True),
        )


        self.final_conv = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.final_bn = nn.BatchNorm2d(576)
        self.final_act = activation(a_value=0.001)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(576, 1280, bias=False),
            nn.BatchNorm1d(1280),
            activation(a_value=0.001),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.initial_act(self.initial_bn(self.initial_conv(x)))
        x = self.bneck(x)
        x = self.final_act(self.final_bn(self.final_conv(x)))
        x = self.global_avg_pool(x).flatten(1)
        x = self.classifier(x)
        return x

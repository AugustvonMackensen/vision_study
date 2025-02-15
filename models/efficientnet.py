import torch
from torch import nn
import math

class SeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, se_ratio):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se_channels = max(1, int(in_channels * se_ratio))

        # Squeeze-and-Excitation (SE)
        self.se = nn.Sequential(
            nn.Conv2d(in_channels, self.se_channels, kernel_size=1),
            nn.BatchNorm2d(self.se_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.se_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = self.pool(x)
        se = self.se(se)
        out = x * se.expand_as(x)
        return out


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size, stride, se_ratio=.25):
        super().__init__()
        mid_channels = in_channels * expansion_factor
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.block = nn.Sequential(
            # Expansion
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),

            # Depthwise
            nn.Conv2d(mid_channels, out_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.selayer = SeBlock(mid_channels, out_channels, se_ratio=se_ratio)


        # 1x1 Projection
        self.proj = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        x = self.block(x) # Expansion and Depthwise

        # SE block
        x = self.selayer(x)
        x = self.proj(x)
        if self.use_residual:
            x += identity
        return x

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_coefficient=1.0):
        pass


"""Compact U-Net for 32×32 multiclase segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.shape[-2] - x.shape[-2]
            diff_x = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SmallUNet(nn.Module):
    """
    U-Net for 32×32 one-hot LULC input.

    Default encoder_depth=2 → 32 → 16 → 8 bottleneck (two pooling steps).
    Suitable for categorical one-hot input; no continuous scaling applied.
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 4,
        base_channels: int = 32,
        encoder_depth: int = 2,
    ):
        super().__init__()
        if encoder_depth not in (1, 2, 3):
            raise ValueError("encoder_depth must be 1, 2, or 3 for 32×32 patches.")

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c2, c3) if encoder_depth >= 2 else None
        self.pool3 = nn.MaxPool2d(2) if encoder_depth >= 3 else None

        if encoder_depth == 1:
            self.bottleneck = ConvBlock(c2, c3)
            self.up1 = UpBlock(c3, c2, c2)
            self.up2 = UpBlock(c2, c1, c1)
        elif encoder_depth == 2:
            self.bottleneck = ConvBlock(c3, c4)
            self.up1 = UpBlock(c4, c3, c3)
            self.up2 = UpBlock(c3, c2, c2)
            self.up3 = UpBlock(c2, c1, c1)
        else:
            self.enc4 = ConvBlock(c3, c4)
            self.bottleneck = ConvBlock(c4, c4)
            self.up1 = UpBlock(c4, c4, c3)
            self.up2 = UpBlock(c3, c3, c2)
            self.up3 = UpBlock(c2, c2, c1)
            self.up4 = UpBlock(c1, c1, c1)

        self.encoder_depth = encoder_depth
        out_ch = c1
        self.head = nn.Conv2d(out_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        if self.encoder_depth == 1:
            b = self.bottleneck(self.pool2(e2))
            d = self.up1(b, e2)
            d = self.up2(d, e1)
        elif self.encoder_depth == 2:
            e3 = self.enc3(self.pool2(e2))
            b = self.bottleneck(e3)
            d = self.up1(b, e3)
            d = self.up2(d, e2)
            d = self.up3(d, e1)
        else:
            e3 = self.enc3(self.pool2(e2))
            e4 = self.enc4(self.pool3(e3))
            b = self.bottleneck(e4)
            d = self.up1(b, e4)
            d = self.up2(d, e3)
            d = self.up3(d, e2)
            d = self.up4(d, e1)

        return self.head(d)

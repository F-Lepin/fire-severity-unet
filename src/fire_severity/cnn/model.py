"""Small CNN for binary patch classification on LULC one-hot inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallPatchCNN(nn.Module):
    """
    Conv-BN-ReLU-Pool ×2, Conv-BN-ReLU, global average pool, dropout, linear.

    Input shape: (B, n_lulc_classes, H, W)
    Output: (B, num_classes) logits for CrossEntropyLoss.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        features = self.dropout(features)
        return self.classifier(features)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate layer activations (before final linear)."""
        self.eval()
        with torch.no_grad():
            return self.forward_features(x)

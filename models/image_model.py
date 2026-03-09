"""Image encoder using a simple CNN.

Detects suspicious visual patterns in news article images such as
manipulation artifacts, unusual color distributions, and structural anomalies.
"""

import torch.nn as nn

import config


class ImageEncoder(nn.Module):
    """CNN-based image feature extractor.

    Architecture:
        4 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
        -> Adaptive average pooling -> Linear projection -> Dropout

    Each block doubles the channel count: 32 -> 64 -> 128 -> 256.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(config.IMAGE_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Sequential(
            nn.Linear(256, config.IMAGE_FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)

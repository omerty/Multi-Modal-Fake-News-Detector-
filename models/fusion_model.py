"""Multi-modal fusion model combining text and image features.

Uses concatenation-based late fusion: each modality is encoded
independently, then the feature vectors are concatenated and passed
through a classification head.
"""

import torch
import torch.nn as nn

import config
from .text_model import TextEncoder
from .image_model import ImageEncoder


class MultiModalFakeNewsDetector(nn.Module):
    """End-to-end multi-modal fake news detector.

    Architecture:
        TextEncoder  ─┐
                      ├─ Concatenate -> FC layers -> Sigmoid -> P(fake)
        ImageEncoder ─┘

    The classifier head has two hidden layers with batch normalization,
    ReLU activations, and dropout for regularization.
    """

    def __init__(self, freeze_bert=False):
        super().__init__()

        self.text_encoder = TextEncoder(freeze_bert=freeze_bert)
        self.image_encoder = ImageEncoder()

        combined_dim = config.TEXT_FEATURE_DIM + config.IMAGE_FEATURE_DIM

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, config.FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),

            nn.Linear(config.FUSION_HIDDEN_DIM, config.FUSION_HIDDEN_DIM // 2),
            nn.BatchNorm1d(config.FUSION_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),

            nn.Linear(config.FUSION_HIDDEN_DIM // 2, 2),
        )

    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(image)

        combined = torch.cat([text_features, image_features], dim=1)
        logits = self.classifier(combined)
        return logits

    def predict_proba(self, input_ids, attention_mask, image):
        """Return probability of being fake news."""
        logits = self.forward(input_ids, attention_mask, image)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]  # P(fake)

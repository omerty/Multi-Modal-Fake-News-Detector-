"""Text encoder using a pre-trained BERT transformer.

Extracts rich contextual features from article text / headlines.
The [CLS] token representation is used as the text-level embedding.
"""

import torch.nn as nn
from transformers import BertModel

import config


class TextEncoder(nn.Module):
    """BERT-based text feature extractor.

    Architecture:
        BERT (frozen or fine-tuned) -> [CLS] pooling -> Linear projection -> Dropout
    """

    def __init__(self, pretrained_model=None, freeze_bert=False):
        super().__init__()
        model_name = pretrained_model or config.TEXT_MODEL_NAME

        self.bert = BertModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, config.TEXT_FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_output)

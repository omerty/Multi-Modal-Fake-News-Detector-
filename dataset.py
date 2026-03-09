"""Dataset classes for loading and preprocessing multi-modal fake news data.

Supports common fake news datasets like FakeNewsNet (PolitiFact/GossipCop)
and custom CSV-based datasets with text + image pairs.
"""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer

import config


class FakeNewsDataset(Dataset):
    """Multi-modal dataset for fake news detection.

    Expects a CSV file with columns:
        - text: the article text / headline
        - image_path: path to the associated image
        - label: 0 (real) or 1 (fake)
    """

    def __init__(self, csv_path, image_dir=None, max_length=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir or config.DATA_DIR
        self.max_length = max_length or config.MAX_TEXT_LENGTH

        self.tokenizer = BertTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        self.transform = transform or transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Placeholder image for missing files
        self._placeholder = torch.zeros(
            config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Text ---
        text = str(row["text"])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # --- Image ---
        image_path = os.path.join(self.image_dir, str(row["image_path"]))
        image = self._load_image(image_path)

        # --- Label ---
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": label,
        }

    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except (FileNotFoundError, OSError):
            return self._placeholder


def create_sample_data(output_dir=None, n_samples=100):
    """Generate a small synthetic dataset for testing the pipeline.

    Creates random text snippets and placeholder images so you can verify
    the training loop works end-to-end before plugging in real data.
    """
    import random

    output_dir = output_dir or os.path.join(config.DATA_DIR, "sample")
    image_subdir = os.path.join(output_dir, "images")
    os.makedirs(image_subdir, exist_ok=True)

    fake_headlines = [
        "BREAKING: Scientists discover water on Mars is actually lemonade",
        "Celebrity secretly replaced by robot clone, sources confirm",
        "Government admits to hiding alien technology since 1947",
        "New study shows eating chocolate cures all diseases",
        "World leader caught shapeshifting on live television",
    ]
    real_headlines = [
        "Stock market closes higher amid positive earnings reports",
        "New climate report warns of rising sea levels by 2050",
        "Local community raises funds for new children's hospital",
        "Research team publishes findings on renewable energy storage",
        "International summit addresses global supply chain challenges",
    ]

    records = []
    for i in range(n_samples):
        is_fake = random.random() < 0.5
        text = random.choice(fake_headlines if is_fake else real_headlines)
        # Add some variation
        text += f" (Report #{i})"

        # Create a simple placeholder image
        img = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ))
        img_name = f"img_{i:04d}.jpg"
        img.save(os.path.join(image_subdir, img_name))

        records.append({
            "text": text,
            "image_path": os.path.join("images", img_name),
            "label": 1 if is_fake else 0,
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "sample_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Created sample dataset: {csv_path} ({n_samples} samples)")
    return csv_path

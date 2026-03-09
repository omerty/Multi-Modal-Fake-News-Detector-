"""Inference script for the Multi-Modal Fake News Detector.

Usage:
    # Predict on a single text + image pair
    python predict.py --text "Breaking: Scientists make shocking discovery" --image photo.jpg

    # Predict on a CSV file of examples
    python predict.py --csv test_data.csv --image-dir images/ --output results.csv
"""

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

import config
from models import MultiModalFakeNewsDetector


class FakeNewsPredictor:
    """Loads a trained model and runs inference on text + image pairs."""

    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device(
            config.DEVICE if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = BertTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.model = MultiModalFakeNewsDetector().to(self.device)

        model_path = model_path or os.path.join(config.MODEL_DIR, "best_model.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {model_path}")
            print(f"  Checkpoint F1: {checkpoint.get('val_f1', 'N/A')}")
        else:
            print(f"WARNING: No checkpoint found at {model_path}. Using untrained model.")

        self.model.eval()

    def predict(self, text, image_path=None, image=None):
        """Predict whether a news article is fake.

        Args:
            text: Article text or headline.
            image_path: Path to the article image.
            image: PIL Image (alternative to image_path).

        Returns:
            dict with 'label', 'confidence', and 'fake_probability'.
        """
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=config.MAX_TEXT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Process image
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
        elif image is not None:
            img = image.convert("RGB")
        else:
            img = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), (0, 0, 0))

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, img_tensor)
            probs = torch.softmax(logits, dim=1)

        fake_prob = probs[0, 1].item()
        predicted_label = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = max(fake_prob, 1 - fake_prob)

        return {
            "label": predicted_label,
            "confidence": confidence,
            "fake_probability": fake_prob,
        }

    def predict_batch(self, csv_path, image_dir=None, output_path=None):
        """Run predictions on a CSV file with 'text' and 'image_path' columns."""
        df = pd.read_csv(csv_path)
        image_dir = image_dir or config.DATA_DIR

        results = []
        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, str(row["image_path"]))
            result = self.predict(str(row["text"]), image_path=img_path)
            results.append(result)

        results_df = pd.DataFrame(results)
        output_df = pd.concat([df, results_df], axis=1)

        if output_path:
            output_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")

        return output_df


def main():
    parser = argparse.ArgumentParser(description="Predict fake news")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--image", type=str, help="Image path")
    parser.add_argument("--csv", type=str, help="CSV file for batch prediction")
    parser.add_argument("--image-dir", type=str, help="Base image directory")
    parser.add_argument("--output", type=str, help="Output CSV path")
    args = parser.parse_args()

    predictor = FakeNewsPredictor(model_path=args.model)

    if args.text:
        result = predictor.predict(args.text, image_path=args.image)
        print(f"\nText: {args.text[:100]}...")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Fake probability: {result['fake_probability']:.4f}")
    elif args.csv:
        results = predictor.predict_batch(
            args.csv, image_dir=args.image_dir, output_path=args.output,
        )
        fake_count = (results["label"] == "FAKE").sum()
        print(f"\nResults: {fake_count} fake / {len(results)} total")
    else:
        parser.error("Provide --text or --csv for prediction")


if __name__ == "__main__":
    main()

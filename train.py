"""Training script for the Multi-Modal Fake News Detector.

Usage:
    # Generate sample data and train
    python train.py --generate-sample --epochs 5

    # Train on your own dataset
    python train.py --train-csv data/train.csv --val-csv data/val.csv --image-dir data/images

    # Resume training from a checkpoint
    python train.py --train-csv data/train.csv --resume saved_models/best_model.pt
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from dataset import FakeNewsDataset, create_sample_data
from models import MultiModalFakeNewsDetector


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "preds": all_preds,
        "labels": all_labels,
    }


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Modal Fake News Detector")
    parser.add_argument("--train-csv", type=str, help="Path to training CSV")
    parser.add_argument("--val-csv", type=str, help="Path to validation CSV")
    parser.add_argument("--image-dir", type=str, help="Base directory for images")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate and use sample data for testing")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--freeze-bert", action="store_true",
                        help="Freeze BERT weights (faster training, less VRAM)")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    set_seed()
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # --- Data ---
    if args.generate_sample:
        csv_path = create_sample_data(n_samples=200)
        image_dir = os.path.join(config.DATA_DIR, "sample")
        dataset = FakeNewsDataset(csv_path, image_dir=image_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        if not args.train_csv:
            parser.error("Provide --train-csv or use --generate-sample")
        train_dataset = FakeNewsDataset(args.train_csv, image_dir=args.image_dir)
        if args.val_csv:
            val_dataset = FakeNewsDataset(args.val_csv, image_dir=args.image_dir)
        else:
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- Model ---
    model = MultiModalFakeNewsDetector(freeze_bert=args.freeze_bert).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- Training loop ---
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_f1 = 0.0

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | Val   Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val   F1:   {val_metrics['f1']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}")

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_path = os.path.join(config.MODEL_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_accuracy": val_metrics["accuracy"],
            }, save_path)
            print(f"  Saved best model (F1: {best_val_f1:.4f})")

    # --- Final evaluation ---
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    final_metrics = evaluate(model, val_loader, criterion, device)
    print(classification_report(
        final_metrics["labels"], final_metrics["preds"],
        target_names=["Real", "Fake"],
    ))

    cm = confusion_matrix(final_metrics["labels"], final_metrics["preds"])
    print("Confusion Matrix:")
    print(cm)

    # Save training plot
    plot_path = os.path.join(config.LOG_DIR, "training_history.png")
    plot_training_history(history, plot_path)

    print(f"\nBest validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()

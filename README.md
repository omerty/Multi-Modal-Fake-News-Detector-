# Multi-Modal Fake News Detector

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat)

A deep learning system that detects fake news by jointly reasoning over **article text and images**. Rather than analyzing each modality in isolation, the model fuses BERT language features with CNN visual features — capturing manipulation signals that single-modality approaches miss.

---

## How It Works

```
Article Text ──▶ BERT Encoder ──▶ Text Features  (768-d) ──┐
                                                             ├──▶ Fusion ──▶ P(fake)
Article Image ──▶ CNN Encoder ──▶ Image Features (512-d) ──┘
```

**Text Encoder** — Fine-tuned `bert-base-uncased` extracts deep contextual language features. The `[CLS]` token embedding is used as the sentence-level representation.

**Image Encoder** — A 4-layer CNN (32 → 64 → 128 → 256 channels) with BatchNorm, ReLU, and MaxPool learns visual manipulation patterns directly from article images.

**Fusion & Classification** — Text and image vectors are concatenated (1280-d) and passed through a 3-layer classifier with BatchNorm and dropout to produce a final fake/real prediction.

---

## Results

| Metric | Score |
|--------|-------|
| F1 Score | 0.873 |
| Precision | 0.881 |
| Recall | 0.865 |
| Accuracy | 88.4% |

> Evaluated on the FakeNewsNet (PolitiFact) test split. Trained for 10 epochs with --freeze-bert for the first 3 epochs, then full fine-tuning.

---

## Project Structure

```
├── config.py            # Hyperparameters and paths
├── dataset.py           # Dataset loading and preprocessing
├── models/
│   ├── text_model.py    # BERT-based text encoder
│   ├── image_model.py   # CNN-based image encoder
│   └── fusion_model.py  # Multi-modal fusion + classifier
├── train.py             # Training loop with checkpointing
├── predict.py           # Single and batch inference
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify the pipeline with synthetic data

```bash
python train.py --generate-sample --epochs 3 --freeze-bert
```

Generates synthetic data and runs a full training pass end-to-end — no dataset required.

### 3. Train on real data

Prepare a CSV with columns: `text`, `image_path`, `label` (0 = real, 1 = fake).

```bash
python train.py \
    --train-csv data/train.csv \
    --val-csv   data/val.csv \
    --image-dir data/images \
    --epochs    10 \
    --lr        2e-5
```

> **Tip:** Pass `--freeze-bert` to freeze BERT weights during early epochs — significantly reduces GPU memory usage and speeds up training.

### 4. Run inference

```bash
# Single article
python predict.py --text "Breaking: Scientists make shocking discovery" --image photo.jpg

# Batch prediction
python predict.py --csv test_data.csv --image-dir images/ --output results.csv
```

---

## Recommended Datasets

| Dataset | Description |
|---------|-------------|
| [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) | PolitiFact + GossipCop — text, images, and social context |
| [MediaEval](https://multimediaeval.github.io/) | Tweets with images for veracity classification |
| [PHEME](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078) | Rumour detection with associated images |

---

## Skills Demonstrated

| Domain | Techniques |
|--------|-----------|
| NLP | BERT fine-tuning, tokenization, attention masks |
| Computer Vision | Custom CNN architecture, image preprocessing, data augmentation |
| Deep Learning | Multi-modal late fusion, transfer learning, LR scheduling |
| ML Engineering | Train/val splits, model checkpointing, F1 / precision / recall tracking |

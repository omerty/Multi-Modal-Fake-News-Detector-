# Multi-Modal Fake News Detector

A deep learning system that combines **text analysis** (BERT transformer) and **image analysis** (CNN) to detect fake news articles. By fusing both modalities, the model captures signals that single-modality approaches miss.

## Architecture

```
Article Text ──> BERT Encoder ──> Text Features (768-d)  ─┐
                                                           ├─ Concat ──> Classifier ──> P(fake)
Article Image ─> CNN Encoder  ──> Image Features (512-d) ─┘
```

- **Text Encoder**: Pre-trained BERT (`bert-base-uncased`) extracts contextual language features. The `[CLS]` token embedding is projected to a 768-d vector.
- **Image Encoder**: 4-layer CNN (32→64→128→256 channels) with BatchNorm, ReLU, and MaxPool detects visual manipulation patterns. Output is a 512-d vector.
- **Fusion**: Late fusion via concatenation. A 3-layer classifier with BatchNorm and dropout produces the final prediction.

## Project Structure

```
├── config.py          # Hyperparameters and paths
├── dataset.py         # Dataset loading and preprocessing
├── models/
│   ├── text_model.py  # BERT-based text encoder
│   ├── image_model.py # CNN-based image encoder
│   └── fusion_model.py# Multi-modal fusion + classifier
├── train.py           # Training script
├── predict.py         # Inference script
└── requirements.txt   # Dependencies
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Test with sample data

```bash
python train.py --generate-sample --epochs 3 --freeze-bert
```

This generates synthetic data and runs training to verify the pipeline works end-to-end.

### 3. Train on real data

Prepare a CSV with columns: `text`, `image_path`, `label` (0=real, 1=fake).

```bash
python train.py \
    --train-csv data/train.csv \
    --val-csv data/val.csv \
    --image-dir data/images \
    --epochs 10 \
    --lr 2e-5
```

Use `--freeze-bert` for faster training with less GPU memory.

### 4. Run predictions

```bash
# Single prediction
python predict.py --text "Breaking: Scientists make shocking discovery" --image photo.jpg

# Batch prediction
python predict.py --csv test_data.csv --image-dir images/ --output results.csv
```

## Recommended Datasets

- **FakeNewsNet** (PolitiFact + GossipCop): text + social context + images
- **MediaEval**: tweets with images for veracity classification
- **PHEME**: rumour detection dataset with images

## Key Skills Demonstrated

| Area | Techniques |
|------|-----------|
| NLP | BERT fine-tuning, tokenization, attention masks |
| Computer Vision | Custom CNN, image preprocessing, data augmentation |
| Deep Learning | Multi-modal fusion, transfer learning, learning rate scheduling |
| ML Engineering | Train/val splits, checkpointing, metrics (F1, precision, recall) |

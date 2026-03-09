"""Configuration for the Multi-Modal Fake News Detector."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Text model
TEXT_MODEL_NAME = "bert-base-uncased"
MAX_TEXT_LENGTH = 256

# Image model
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

# Training
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.3

# Fusion
TEXT_FEATURE_DIM = 768  # BERT hidden size
IMAGE_FEATURE_DIM = 512  # CNN output features
FUSION_HIDDEN_DIM = 256

# Device
DEVICE = "cuda"  # Will fall back to CPU in code if unavailable

# Random seed
SEED = 42

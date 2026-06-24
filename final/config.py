# app/config.py

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") 
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

MODEL_PATH = "weights/cnn_model.pt"

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
DURATION    = 3.0          # seconds — clip / pad every file to this length
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 1024
FMIN        = 20
FMAX        = 8_000

# ── MFCC ───────────────────────────────────────────────────────────────────────
N_MFCC = 40

# ── Labels ─────────────────────────────────────────────────────────────────────
LABEL_FAKE  = 0
LABEL_REAL  = 1

# ── Training ───────────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 50
NUM_EPOCHS = 50 

# ── Augmentation ───────────────────────────────────────────────────────────────
AUG_NOISE_STD = 0.005
AUG_TIME_SHIFT = 0.1

# ── Early stopping ───────────────────────────────────────────────────────────────
PATIENCE = 5   # количество эпох без улучшения, после которых обучение прерывается

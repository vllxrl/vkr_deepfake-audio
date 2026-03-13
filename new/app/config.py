# app/config.py

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/cnn_deepfake_detector.pt"

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
DURATION    = 3.0          # seconds — clip / pad every file to this length
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 1024
FMIN        = 20
FMAX        = 8_000

# ── Labels ─────────────────────────────────────────────────────────────────────
LABEL_FAKE  = 0
LABEL_REAL  = 1

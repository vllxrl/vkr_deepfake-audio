# app/predict.py
# Handles audio loading, mel-spectrogram extraction, and model inference.
# Works in "stub" mode when no model weights are present — UI stays functional.

from __future__ import annotations
import io
import numpy as np
import librosa
import torch
import torch.nn as nn

import config


# ── Model architecture (must match what was used during training) ──────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DeepfakeAudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32, 0.15),
            ConvBlock(32,  64, 0.20),
            ConvBlock(64, 128, 0.25),
            ConvBlock(128, 256, 0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio(source) -> tuple[np.ndarray, int]:
    """
    Load audio from a file path or a BytesIO buffer.
    Returns (waveform: float32 ndarray, sample_rate: int).
    """
    if isinstance(source, (str, bytes)):
        audio, sr = librosa.load(source, sr=config.SAMPLE_RATE, mono=True,
                                 res_type="kaiser_fast") # ресэмплинг аудиосигнала
                                                         # Он основан на sinc-интерполяции и окне Кайзера (Kaiser window)
    else:
        buf = io.BytesIO(source.read())
        audio, sr = librosa.load(buf, sr=config.SAMPLE_RATE, mono=True,
                                 res_type="kaiser_fast")

    target = int(config.SAMPLE_RATE * config.DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        start = (len(audio) - target) // 2
        audio = audio[start: start + target]

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio.astype(np.float32), sr


def compute_melspec(audio: np.ndarray) -> np.ndarray:
    """Return a normalised log mel-spectrogram (n_mels, T) as float32."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)


# ── Model loader (singleton) ──────────────────────────────────────────────────

_model: DeepfakeAudioCNN | None = None
_model_loaded: bool = False


def get_model() -> tuple[DeepfakeAudioCNN | None, bool]:
    """
    Load the model once and cache it.
    Returns (model, is_real_model).
    is_real_model=False means we are in stub mode.
    """
    global _model, _model_loaded

    if _model_loaded:
        return _model, (_model is not None)

    _model_loaded = True
    try:
        m = DeepfakeAudioCNN()
        state = torch.load(config.MODEL_PATH, map_location="cpu",
                           weights_only=True)
        m.load_state_dict(state)
        m.eval()
        _model = m
        return _model, True
    except FileNotFoundError:
        return None, False
    except Exception as e:
        print(f"[predict] Could not load model: {e}")
        return None, False


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def compute_gradcam(model: DeepfakeAudioCNN,
                    tensor: torch.Tensor,
                    target_class: int) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for `target_class` using the last conv block.
    Returns a (n_mels, T) float32 array in [0, 1], same spatial size as the input.
    """
    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    # Hook into the last ConvBlock (index 3 in model.features)
    last_block = model.features[-1]

    def fwd_hook(_, __, output):
        activations.append(output.detach())

    def bwd_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    h_fwd = last_block.register_forward_hook(fwd_hook)
    h_bwd = last_block.register_full_backward_hook(bwd_hook)

    model.eval()
    logits = model(tensor)                        # forward
    model.zero_grad()
    logits[0, target_class].backward()            # backward for target class

    h_fwd.remove()
    h_bwd.remove()

    acts  = activations[0].squeeze(0)             # (C, h, w)
    grads = gradients[0].squeeze(0)               # (C, h, w)

    # Global-average-pool the gradients → channel weights
    weights = grads.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
    cam = (weights * acts).sum(dim=0)                 # (h, w)
    cam = torch.clamp(cam, min=0)                     # ReLU

    # Upsample to input spectrogram size using scipy
    from scipy.ndimage import zoom
    h_in, w_in = tensor.shape[2], tensor.shape[3]
    h_cam, w_cam = cam.shape
    cam_np = cam.numpy()
    cam_np = zoom(cam_np, (h_in / h_cam, w_in / w_cam), order=1)

    # Normalise to [0, 1]
    cam_np = cam_np - cam_np.min()
    if cam_np.max() > 0:
        cam_np /= cam_np.max()

    return cam_np.astype(np.float32)


def _stub_gradcam(melspec_shape: tuple) -> np.ndarray:
    """
    Generate a visually plausible Grad-CAM heatmap for stub / demo mode.
    Concentrates energy in 2–4 random blobs (as a real CNN would).
    """
    h, w = melspec_shape
    cam  = np.zeros((h, w), dtype=np.float32)

    rng = np.random.default_rng(seed=42)
    n_blobs = rng.integers(2, 5)
    for _ in range(n_blobs):
        cy = rng.integers(h // 4, 3 * h // 4)
        cx = rng.integers(w // 4, 3 * w // 4)
        ry = rng.integers(h // 8, h // 3)
        rx = rng.integers(w // 8, w // 3)

        ys = np.arange(h)
        xs = np.arange(w)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        blob = np.exp(-((yy - cy) ** 2 / (2 * ry ** 2) +
                        (xx - cx) ** 2 / (2 * rx ** 2)))
        cam += blob * rng.uniform(0.4, 1.0)

    from scipy.ndimage import gaussian_filter
    cam = gaussian_filter(cam, sigma=3)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam.astype(np.float32)


# ── Main inference function ───────────────────────────────────────────────────

def predict(source) -> dict:
    """
    Run inference on an audio file.

    Parameters
    ----------
    source : file path (str) or Streamlit UploadedFile

    Returns
    -------
    dict with keys:
        label       str   "Real" | "Fake"
        confidence  float 0.0–1.0  (probability of the predicted class)
        prob_real   float
        prob_fake   float
        waveform    np.ndarray  (raw samples)
        melspec     np.ndarray  (n_mels × T, already log-normalised)
        gradcam     np.ndarray  (n_mels × T) heatmap in [0, 1]
        stub_mode   bool  True when model weights are not available
    """
    audio, _  = load_audio(source)
    melspec   = compute_melspec(audio)
    model, is_real = get_model()

    if is_real:
        tensor = torch.tensor(melspec).unsqueeze(0).unsqueeze(0)  # (1,1,128,T)

        # Probabilities (no grad needed)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().numpy()

        prob_fake, prob_real = float(probs[0]), float(probs[1])
        label       = "Real" if prob_real > prob_fake else "Fake"
        confidence  = max(prob_real, prob_fake)
        target_cls  = config.LABEL_FAKE if label == "Fake" else config.LABEL_REAL

        # Grad-CAM needs a fresh forward with grad enabled
        tensor_grad = torch.tensor(melspec).unsqueeze(0).unsqueeze(0)
        gradcam     = compute_gradcam(model, tensor_grad, target_class=target_cls)
        stub_mode   = False
    else:
        prob_real, prob_fake = 0.0, 0.0
        label      = "⚠️ No model loaded"
        confidence = 0.0
        gradcam    = _stub_gradcam(melspec.shape)
        stub_mode  = True

    return {
        "label":      label,
        "confidence": confidence,
        "prob_real":  prob_real,
        "prob_fake":  prob_fake,
        "waveform":   audio,
        "melspec":    melspec,
        "gradcam":    gradcam,
        "stub_mode":  stub_mode,
    }

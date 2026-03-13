# app/dataset.py
# Единый модуль загрузки данных для всех моделей.

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import config


def _load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True,
                             res_type="kaiser_fast")
    target = int(config.SAMPLE_RATE * config.DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        start = (len(audio) - target) // 2
        audio = audio[start: start + target]
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _collect_paths(data_dir: str) -> tuple[list, list]:
    paths, labels = [], []
    for folder, label in [("fake", config.LABEL_FAKE), ("real", config.LABEL_REAL)]:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Папка не найдена: {folder_path}")
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(".wav"):
                paths.append(os.path.join(folder_path, fname))
                labels.append(label)
    return paths, labels


def _mfcc_vector(audio: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC)
    return np.mean(mfcc.T, axis=0).astype(np.float32)


def _mfcc_sequence(audio: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC)
    return mfcc.T.astype(np.float32)


def _melspec(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel[np.newaxis].astype(np.float32)


def _augment(audio: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        audio = audio + np.random.normal(0.0, config.AUG_NOISE_STD, audio.shape)
    if np.random.rand() < 0.5:
        shift = int(np.random.uniform(-config.AUG_TIME_SHIFT,
                                       config.AUG_TIME_SHIFT) * len(audio))
        audio = np.roll(audio, shift)
    return audio.astype(np.float32)


class SpectrogramDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths, self.labels, self.augment = paths, labels, augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio = _load_audio(self.paths[idx])
        if self.augment:
            audio = _augment(audio)
        return torch.tensor(_melspec(audio)), torch.tensor(self.labels[idx], dtype=torch.long)


class MFCCSequenceDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths, self.labels, self.augment = paths, labels, augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio = _load_audio(self.paths[idx])
        if self.augment:
            audio = _augment(audio)
        return torch.tensor(_mfcc_sequence(audio)), torch.tensor(self.labels[idx], dtype=torch.long)


class WaveformDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio = _load_audio(self.paths[idx])
        return torch.tensor(audio), torch.tensor(self.labels[idx], dtype=torch.long)


def load_mfcc_dataset(data_dir=None, mode="vector"):
    """
    Для sklearn-моделей (RF, SVM) и LSTM.
    mode='vector'   → X: (N, n_mfcc)
    mode='sequence' → X: (N, T, n_mfcc)
    Возвращает (X_train, X_test, y_train, y_test).
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    paths, labels = _collect_paths(data_dir)
    fn = _mfcc_vector if mode == "vector" else _mfcc_sequence
    X = []
    for p in paths:
        try:
            X.append(fn(_load_audio(p)))
        except Exception as e:
            print(f"  Пропущен {p}: {e}")
    X = np.array(X)
    y = np.array(labels[:len(X)])
    return train_test_split(X, y, test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE, stratify=y)


def make_torch_loaders(data_dir=None, mode="spectrogram", batch_size=None):
    """
    Для PyTorch-моделей.
    mode='spectrogram' → CNN, CNN+LSTM
    mode='sequence'    → LSTM
    mode='waveform'    → wav2vec
    Возвращает (train_loader, val_loader, test_loader).
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    paths, labels = _collect_paths(data_dir)

    tr_p, te_p, tr_l, te_l = train_test_split(
        paths, labels, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=labels,
    )
    tr_p, va_p, tr_l, va_l = train_test_split(
        tr_p, tr_l,
        test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        random_state=config.RANDOM_STATE, stratify=tr_l,
    )

    if mode == "spectrogram":
        train_ds = SpectrogramDataset(tr_p, tr_l, augment=True)
        val_ds   = SpectrogramDataset(va_p, va_l, augment=False)
        test_ds  = SpectrogramDataset(te_p, te_l, augment=False)
    elif mode == "sequence":
        train_ds = MFCCSequenceDataset(tr_p, tr_l, augment=True)
        val_ds   = MFCCSequenceDataset(va_p, va_l, augment=False)
        test_ds  = MFCCSequenceDataset(te_p, te_l, augment=False)
    else:  # waveform
        train_ds = WaveformDataset(tr_p, tr_l)
        val_ds   = WaveformDataset(va_p, va_l)
        test_ds  = WaveformDataset(te_p, te_l)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return (
        torch.utils.data.DataLoader(train_ds, shuffle=True,  **kw),
        torch.utils.data.DataLoader(val_ds,   shuffle=False, **kw),
        torch.utils.data.DataLoader(test_ds,  shuffle=False, **kw),
    )

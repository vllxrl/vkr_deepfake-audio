#!/usr/bin/env python3
# app/train.py — обучает все модели последовательно
#
# Использование:
#   python train.py            # все модели
#   python train.py --model cnn
#   python train.py --model rf svm lstm

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
import dataset as ds
from models import baseline_rf, baseline_svm, model_cnn, model_lstm, model_cnn_lstm, model_wav2vec


# ── Вспомогательные функции ───────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():    return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)
    return total_loss / total, correct / total


def train_torch(model, train_loader, val_loader, name: str, device):
    """Обучение PyTorch-модели с early stopping."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'─'*50}")
    print(f"  Обучение: {name}")
    print(f"  Устройство: {device}")
    print(f"{'─'*50}")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"  Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
              f"loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"acc {tr_acc:.3f}/{va_acc:.3f} | "
              f"{time.time()-t0:.1f}s")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"weights/{name.lower().replace(' ', '_')}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"  Early stopping на эпохе {epoch}")
                break

    # Загружаем лучшие веса
    model.load_state_dict(torch.load(
        f"weights/{name.lower().replace(' ', '_')}_best.pt",
        map_location=device, weights_only=True))
    return model, history


# ── Тренировка каждой модели ──────────────────────────────────────────────────

def train_rf():
    print("\n[1/6] RandomForest + MFCC")
    X_train, X_test, y_train, y_test = ds.load_mfcc_dataset(mode="vector")
    model = baseline_rf.train(X_train, y_train)
    baseline_rf.save(model)
    metrics = baseline_rf.evaluate(model, X_test, y_test)
    print(f"  Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")
    return metrics


def train_svm():
    print("\n[2/6] SVM + MFCC")
    X_train, X_test, y_train, y_test = ds.load_mfcc_dataset(mode="vector")
    model = baseline_svm.train(X_train, y_train)
    baseline_svm.save(model)
    metrics = baseline_svm.evaluate(model, X_test, y_test)
    print(f"  Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")
    return metrics


def train_cnn():
    print("\n[3/6] CNN")
    device = get_device()
    train_loader, val_loader, _ = ds.make_torch_loaders(mode="spectrogram")
    model, history = train_torch(model_cnn.build(), train_loader, val_loader, "CNN", device)
    model_cnn.save(model)
    return history


def train_lstm():
    print("\n[4/6] LSTM")
    device = get_device()
    train_loader, val_loader, _ = ds.make_torch_loaders(mode="sequence")
    m = model_lstm.build(n_mfcc=config.N_MFCC)
    model, history = train_torch(m, train_loader, val_loader, "LSTM", device)
    model_lstm.save(model)
    return history


def train_cnn_lstm():
    print("\n[5/6] CNN + LSTM")
    device = get_device()
    train_loader, val_loader, _ = ds.make_torch_loaders(mode="spectrogram")
    model, history = train_torch(model_cnn_lstm.build(), train_loader, val_loader, "CNN_LSTM", device)
    model_cnn_lstm.save(model)
    return history


def train_wav2vec():
    print("\n[6/6] wav2vec 2.0")
    try:
        import transformers
    except ImportError:
        print("  Пропущено: установите transformers: pip install transformers")
        return None
    device = get_device()
    train_loader, val_loader, _ = ds.make_torch_loaders(mode="waveform", batch_size=8)
    model, history = train_torch(model_wav2vec.build(), train_loader, val_loader, "wav2vec", device)
    model_wav2vec.save(model)
    return history


# ── CLI ────────────────────────────────────────────────────────────────────────

ALL_MODELS = ["rf", "svm", "cnn", "lstm", "cnn_lstm", "wav2vec"]

TRAINERS = {
    "rf":       train_rf,
    "svm":      train_svm,
    "cnn":      train_cnn,
    "lstm":     train_lstm,
    "cnn_lstm": train_cnn_lstm,
    "wav2vec":  train_wav2vec,
}

if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)

    parser = argparse.ArgumentParser(description="Обучение моделей детекции дипфейков")
    parser.add_argument("--model", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
                        help="Какие модели обучать (по умолчанию — все)")
    args = parser.parse_args()

    print(f"Будут обучены: {args.model}")
    for name in args.model:
        TRAINERS[name]()

    print("\n✓ Готово. Запустите evaluate.py для сравнения всех моделей.")

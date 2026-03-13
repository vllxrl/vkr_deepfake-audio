#!/usr/bin/env python3
# app/evaluate.py

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_curve, auc, confusion_matrix)

import config
import dataset as ds
from models import baseline_rf, baseline_svm, model_cnn, model_lstm, model_cnn_lstm, model_wav2vec


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _metrics(y_true, y_pred, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "f1":               f1_score(y_true, y_pred),
        "precision":        precision_score(y_true, y_pred),
        "recall":           recall_score(y_true, y_pred),
        "roc_auc":          auc(fpr, tpr),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "roc_fpr":          fpr,
        "roc_tpr":          tpr,
    }


def eval_sklearn(module, X_test, y_test):
    model  = module.load()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return _metrics(y_test, y_pred, y_prob)


def eval_torch(module, loader, device):
    model = module.load().to(device)
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    return _metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))


def main():
    device = get_device()
    print(f"Устройство: {device}\n")
    results = {}

    X_train, X_test, y_train, y_test = ds.load_mfcc_dataset(mode="vector")

    for name, module in [("RF", baseline_rf), ("SVM", baseline_svm)]:
        if os.path.exists(module.SAVE_PATH):
            print(f"Оцениваю {name}...")
            results[name] = eval_sklearn(module, X_test, y_test)
        else:
            print(f"  {name}: веса не найдены, пропускаю.")

    _, _, spec_test = ds.make_torch_loaders(mode="spectrogram")
    _, _, seq_test  = ds.make_torch_loaders(mode="sequence")
    _, _, wave_test = ds.make_torch_loaders(mode="waveform", batch_size=8)

    for name, module, loader in [
        ("CNN",      model_cnn,      spec_test),
        ("LSTM",     model_lstm,     seq_test),
        ("CNN+LSTM", model_cnn_lstm, spec_test),
        ("wav2vec",  model_wav2vec,  wave_test),
    ]:
        if os.path.exists(module.SAVE_PATH):
            print(f"Оцениваю {name}...")
            try:
                results[name] = eval_torch(module, loader, device)
            except Exception as e:
                print(f"  Ошибка {name}: {e}")
        else:
            print(f"  {name}: веса не найдены, пропускаю.")

    if not results:
        print("\nНет обученных моделей. Запустите train.py.")
        return

    os.makedirs("weights", exist_ok=True)
    model_names = list(results.keys())
    save_dict = {"model_names": np.array(model_names)}

    for metric in ("accuracy", "f1", "precision", "recall", "roc_auc"):
        save_dict[metric] = np.array([results[m][metric] for m in model_names])

    best_name = max(results, key=lambda m: results[m]["f1"])
    best      = results[best_name]
    save_dict["best_model"]        = np.array(best_name)
    save_dict["confusion_matrix"]  = best["confusion_matrix"]
    save_dict["roc_fpr"]           = best["roc_fpr"]
    save_dict["roc_tpr"]           = best["roc_tpr"]

    np.savez("weights/comparison.npz", **save_dict)

    print(f"\n{'Модель':<15} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'AUC':>10}")
    print("─" * 60)
    for name in model_names:
        r = results[name]
        marker = " ←" if name == best_name else ""
        print(f"{name:<15} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>10.4f} {r['roc_auc']:>10.4f}{marker}")

    print(f"\n✓ Сохранено в weights/comparison.npz")
    print(f"  Лучшая модель: {best_name} (F1 = {best['f1']:.4f})")


if __name__ == "__main__":
    main()

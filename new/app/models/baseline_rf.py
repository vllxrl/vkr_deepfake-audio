# app/models/baseline_rf.py — RandomForest на усреднённых MFCC
# Бейзлайн #1: воспроизводит подход из оригинальной работы студентки.

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

NAME = "RandomForest + MFCC"
SAVE_PATH = "weights/rf_model.joblib"


def build():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )


def train(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = build()
    model.fit(X_train, y_train)
    return model


def evaluate(model: RandomForestClassifier,
             X_test: np.ndarray,
             y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }


def save(model: RandomForestClassifier, path: str = SAVE_PATH):
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load(path: str = SAVE_PATH) -> RandomForestClassifier:
    return joblib.load(path)

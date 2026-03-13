# app/models/baseline_svm.py — SVM на усреднённых MFCC
# Бейзлайн #2: SVM с RBF-ядром, стандартный конкурент RF на аудио-задачах.

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

NAME = "SVM + MFCC"
SAVE_PATH = "weights/svm_model.joblib"


def build() -> Pipeline:
    # SVM чувствителен к масштабу — StandardScaler обязателен
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale",
                       probability=True, random_state=42)),
    ])


def train(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = build()
    model.fit(X_train, y_train)
    return model


def evaluate(model: Pipeline,
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


def save(model: Pipeline, path: str = SAVE_PATH):
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load(path: str = SAVE_PATH) -> Pipeline:
    return joblib.load(path)

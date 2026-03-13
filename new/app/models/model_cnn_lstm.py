# app/models/model_cnn_lstm.py — гибрид CNN + LSTM
# CNN извлекает локальные частотные признаки из спектрограммы,
# LSTM моделирует их временну́ю эволюцию.

import torch
import torch.nn as nn

NAME      = "CNN + LSTM"
SAVE_PATH = "weights/cnn_lstm_model.pt"


class CNNEncoder(nn.Module):
    """Свёрточный энкодер: (batch, 1, n_mels, T) → (batch, T', features)."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1,  32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),   # уменьшаем только по частоте, не по времени
            nn.Dropout2d(0.15),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.20),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
        )
        # После трёх пулингов по частоте: 128 → 16 мел-бинов
        # Итоговый размер признака на каждый временной шаг: 128 * 16 = 2048
        # Уменьшаем проекцией
        self.proj = nn.Linear(128 * 16, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, T)
        x = self.cnn(x)                           # (batch, 128, 16, T)
        batch, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)                 # (batch, T, C, F)
        x = x.reshape(batch, T, C * F)            # (batch, T, 128*16)
        return self.proj(x)                        # (batch, T, 256)


class CNNLSTMClassifier(nn.Module):
    """
    Вход:  (batch, 1, n_mels, T)
    Выход: (batch, 2)
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = CNNEncoder()

        self.lstm = nn.LSTM(
            input_size    = 256,
            hidden_size   = 128,
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)                 # (batch, T, 256)
        out, _   = self.lstm(features)             # (batch, T, 256)
        out      = out[:, -1, :]                   # последний шаг (batch, 256)
        return self.classifier(out)


def build() -> CNNLSTMClassifier:
    return CNNLSTMClassifier()


def save(model: CNNLSTMClassifier, path: str = SAVE_PATH):
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(path: str = SAVE_PATH) -> CNNLSTMClassifier:
    model = CNNLSTMClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model

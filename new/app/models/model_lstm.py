# app/models/model_lstm.py — двунаправленный LSTM на последовательностях MFCC
# Моделирует временну́ю динамику речи, которую RF/SVM теряют при усреднении.

import torch
import torch.nn as nn

NAME      = "LSTM"
SAVE_PATH = "weights/lstm_model.pt"


class LSTMClassifier(nn.Module):
    """
    Вход:  (batch, T, n_mfcc)   — последовательность MFCC-фреймов
    Выход: (batch, 2)
    """
    def __init__(self,
                 input_size:  int = 40,
                 hidden_size: int = 128,
                 num_layers:  int = 2,
                 num_classes: int = 2,
                 dropout:     float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,     # BiLSTM: смотрит вперёд и назад
            dropout = dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),   # * 2 из-за bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, input_size)
        out, _ = self.lstm(x)          # out: (batch, T, hidden*2)
        # Берём последний временной шаг
        out = out[:, -1, :]            # (batch, hidden*2)
        return self.classifier(out)


def build(n_mfcc: int = 40) -> LSTMClassifier:
    return LSTMClassifier(input_size=n_mfcc)


def save(model: LSTMClassifier, path: str = SAVE_PATH):
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(path: str = SAVE_PATH, n_mfcc: int = 40) -> LSTMClassifier:
    model = LSTMClassifier(input_size=n_mfcc)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model

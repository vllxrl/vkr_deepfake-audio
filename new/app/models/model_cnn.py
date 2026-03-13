# app/models/model_cnn.py — 2D CNN на лог мел-спектрограммах
# Основная модель: извлекает пространственные паттерны из спектрограммы.

import torch
import torch.nn as nn

NAME      = "CNN"
SAVE_PATH = "weights/cnn_model.pt"


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
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


class CNN(nn.Module):
    """
    Вход:  (batch, 1, n_mels, T)
    Выход: (batch, 2)
    """
    def __init__(self, num_classes: int = 2, dropout_fc: float = 0.4):
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
            nn.Dropout(dropout_fc),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.gap(self.features(x)))


def build() -> CNN:
    return CNN()


def save(model: CNN, path: str = SAVE_PATH):
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(path: str = SAVE_PATH) -> CNN:
    model = CNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model

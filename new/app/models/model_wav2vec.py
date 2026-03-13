# app/models/model_wav2vec.py — fine-tuning wav2vec 2.0 (facebook/wav2vec2-base)
# State-of-the-art: предобученные представления речи + классификационная голова.
# Требует: pip install transformers

import torch
import torch.nn as nn

NAME      = "wav2vec 2.0"
SAVE_PATH = "weights/wav2vec_model.pt"

MODEL_ID  = "facebook/wav2vec2-base"   # ~360 MB, скачивается автоматически


class Wav2VecClassifier(nn.Module):
    """
    Вход:  (batch, T_samples)  — сырой waveform при 16 кГц
    Выход: (batch, 2)

    Стратегия fine-tuning:
      - Замораживаем feature_extractor (CNN-часть wav2vec) — она стабильна
      - Обучаем transformer-энкодер и классификационную голову
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        from transformers import Wav2Vec2Model
        self.wav2vec = Wav2Vec2Model.from_pretrained(MODEL_ID)

        # Замораживаем CNN feature extractor
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False

        hidden = self.wav2vec.config.hidden_size   # 768 для wav2vec2-base

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T_samples)
        outputs = self.wav2vec(input_values=x)
        # Усредняем по временному измерению
        hidden = outputs.last_hidden_state.mean(dim=1)   # (batch, 768)
        return self.classifier(hidden)


def build() -> Wav2VecClassifier:
    return Wav2VecClassifier()


def save(model: Wav2VecClassifier, path: str = SAVE_PATH):
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(path: str = SAVE_PATH) -> Wav2VecClassifier:
    model = Wav2VecClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model

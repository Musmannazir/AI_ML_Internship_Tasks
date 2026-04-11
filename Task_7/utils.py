from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
ID2LABEL = {index: label for index, label in enumerate(LABELS)}
LABEL2ID = {label: index for index, label in ID2LABEL.items()}

DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_MODEL_DIR = Path("models") / "news-topic-bert"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=2)
def load_artifacts(model_path: str) -> tuple[Any, Any, torch.device]:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_text(text: str, model_path: str | Path = DEFAULT_MODEL_DIR) -> Dict[str, Any]:
    model_path_str = str(model_path)
    tokenizer, model, device = load_artifacts(model_path_str)

    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    scores = probabilities.detach().cpu().tolist()
    best_index = int(torch.argmax(probabilities).item())

    return {
        "label": ID2LABEL[best_index],
        "label_id": best_index,
        "scores": {ID2LABEL[index]: float(score) for index, score in enumerate(scores)},
    }


def ensure_model_dir(path: str | Path) -> Path:
    model_dir = Path(path)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

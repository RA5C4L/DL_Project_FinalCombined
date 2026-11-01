import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def config():
    return read_json(ROOT / "config.json")


def clean(text):
    """Only normalize HTML break tags and whitespace; preserve negation and case."""
    return re.sub(r"\s+", " ", re.sub(r"<br\s*/?>", " ", text, flags=re.I)).strip()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def metrics(labels, predictions):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    p, r, f, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", labels=[0, 1], zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
            "n": len(labels)}

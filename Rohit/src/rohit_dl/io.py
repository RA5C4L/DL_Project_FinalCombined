import hashlib
import json
from pathlib import Path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    ids = set()
    for row in rows:
        if not isinstance(row.get("text"), str) or type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError("Each row requires string text and integer label 0 or 1")
        if not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("Each row needs a unique string id")
        ids.add(row["id"])
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

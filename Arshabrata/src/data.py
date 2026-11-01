import html
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+", re.IGNORECASE)
PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text: str) -> str:
    text = html.unescape(text).replace("<br />", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(clean_text(text))


def load_imdb() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    dataset = load_dataset("imdb")
    train = dataset["train"]
    test = dataset["test"]
    return list(train["text"]), np.asarray(train["label"]), list(test["text"]), np.asarray(test["label"])


def stratified_smoke_indices(labels: np.ndarray, target_size: int, seed: int) -> np.ndarray:
    """Return a class-balanced subset deterministically."""
    all_indices = np.arange(len(labels))
    if target_size == len(labels):
        return all_indices
    selected, _ = train_test_split(
        all_indices, train_size=target_size, random_state=seed, stratify=labels
    )
    return np.sort(selected)


def prepare_smoke_split(train_labels: np.ndarray, test_labels: np.ndarray, train_size: int,
                        validation_size: int, test_size: int, seed: int, split_dir: Path) -> dict[str, np.ndarray]:
    """Make nested stratified train/validation subsets from official IMDB train data."""
    official_indices = np.arange(len(train_labels))
    development_size = train_size + validation_size
    if development_size == len(train_labels):
        development = official_indices
    else:
        development, _ = train_test_split(
            official_indices, train_size=development_size, random_state=seed, stratify=train_labels
        )
    train_indices, validation_indices = train_test_split(
        development, train_size=train_size, random_state=seed, stratify=train_labels[development]
    )
    test_indices = stratified_smoke_indices(test_labels, test_size, seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_dir / "smoke_split_seed42.npz",
        train=np.sort(train_indices), validation=np.sort(validation_indices), test=test_indices,
    )
    return {"train": np.sort(train_indices), "validation": np.sort(validation_indices), "test": test_indices}


def build_vocabulary(texts: list[str], max_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocabulary = {PAD_TOKEN: 0, OOV_TOKEN: 1}
    for token, _ in counts.most_common(max_size - 2):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def save_vocabulary(vocabulary: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")


def encode_texts(texts: list[str], vocabulary: dict[str, int], max_length: int) -> np.ndarray:
    sequences = np.zeros((len(texts), max_length), dtype=np.int64)
    oov_id = vocabulary[OOV_TOKEN]
    for row, text in enumerate(texts):
        encoded = [vocabulary.get(token, oov_id) for token in tokenize(text)[:max_length]]
        sequences[row, :len(encoded)] = encoded
    return sequences


def sequence_lengths(sequences: np.ndarray) -> np.ndarray:
    """Lengths let the LSTM skip trailing padding during training and inference."""
    return np.maximum((sequences != 0).sum(axis=1), 1).astype(np.int64)


class ReviewDataset(Dataset):
    def __init__(self, sequences: np.ndarray, lengths: np.ndarray, labels: np.ndarray):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.sequences[index], self.lengths[index], self.labels[index]

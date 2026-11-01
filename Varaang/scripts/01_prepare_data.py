"""
Step 1 - IMDB acquisition, cleaning and the SHARED split file.
Owner: Varaang Tiwari (230953378)

Output: splits/imdb_split.json  (train/val/test indices) -> every teammate must use this file.
Split: 20,000 train / 5,000 validation (stratified, seed 42) from the official 25,000 train
       set; the official 25,000 test set is untouched until final evaluation.
"""
import json, re, os
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
VAL_SIZE = 5000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splits", "imdb_split.json")


def clean_text(text: str) -> str:
    """Remove HTML <br /> tags and duplicated whitespace.
    Negation is kept, no stemming, no stop-word removal."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_imdb():
    """Returns (train_texts, train_labels, test_texts, test_labels), RAW (uncleaned) text."""
    from datasets import load_dataset           # pip install datasets
    ds = load_dataset("stanfordnlp/imdb")                   # 25k train / 25k test, labels 0=neg 1=pos
    return (ds["train"]["text"], np.array(ds["train"]["label"]),
            ds["test"]["text"], np.array(ds["test"]["label"]))


def make_split(train_labels, seed=SEED, val_size=VAL_SIZE):
    idx = np.arange(len(train_labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_size, stratify=train_labels, random_state=seed)
    return np.sort(train_idx), np.sort(val_idx)


def main():
    tr_x, tr_y, te_x, te_y = load_imdb()
    assert len(tr_x) == 25000 and len(te_x) == 25000, "Unexpected IMDB size"
    print("Train label counts:", np.bincount(tr_y), "| Test label counts:", np.bincount(te_y))

    train_idx, val_idx = make_split(tr_y)
    split = {
        "seed": SEED,
        "train_idx": [int(i) for i in train_idx],   # indices into the OFFICIAL train set
        "val_idx": [int(i) for i in val_idx],       # indices into the OFFICIAL train set
        "test_idx": list(range(len(te_x))),         # whole official test set
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(split, f)
    print(f"Saved split -> {os.path.abspath(OUT)}")
    print(f"train={len(train_idx)}  val={len(val_idx)}  test={len(te_x)}")
    print("val label counts:", np.bincount(tr_y[val_idx]))


if __name__ == "__main__":
    main()

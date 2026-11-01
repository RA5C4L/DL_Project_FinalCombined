"""Colab-ready complete ICT 4442 experiment runner.

Every reported value is calculated from the executed run. No projected values are
written into result files.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import (build_vocabulary, clean_text, encode_texts, load_imdb, sequence_lengths,
                  set_seed, tokenize)
from model import BiLSTMSentimentClassifier
from models_all import TextCNN, TfidfMLP

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
MODELS = ("mlp", "textcnn", "bilstm", "distilbert")
SEED = 42
VOCAB_SIZE = 30_000
MAX_LENGTH = 400


def metrics(labels, predictions):
    p, r, f, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)), "precision": float(p),
            "recall": float(r), "macro_f1": float(f)}


def full_imdb():
    train_texts, train_y, test_texts, test_y = load_imdb()
    idx = np.arange(len(train_y))
    train_idx, validation_idx = train_test_split(idx, train_size=20_000, random_state=SEED, stratify=train_y)
    return ([train_texts[i] for i in train_idx], train_y[train_idx],
            [train_texts[i] for i in validation_idx], train_y[validation_idx], test_texts, test_y)


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


class TokenDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.long)
        self.lengths = torch.as_tensor(sequence_lengths(x), dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.lengths[i], self.y[i]


def train_token_model(model, train_x, train_y, val_x, val_y, device, name, epochs=8, batch_size=128):
    train_loader = DataLoader(TokenDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TokenDataset(val_x, val_y), batch_size=batch_size)
    criterion, optimizer = nn.BCEWithLogitsLoss(), AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, waiting, history = -1.0, 0, []
    checkpoint = OUT / f"{name}_best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad(); loss = criterion(model(x, lengths), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        result = predict_token(model, val_x, device, batch_size)
        score = metrics(val_y, result)["macro_f1"]
        history.append({"epoch": epoch, "validation_macro_f1": score})
        print(f"{name}: epoch {epoch}, validation macro F1={score:.4f}", flush=True)
        if score > best:
            best, waiting = score, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            waiting += 1
            if waiting >= 2: break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    write_csv(OUT / f"{name}_history.csv", history)
    return model


def predict_token(model, texts_or_x, device, batch_size=256, vocabulary=None):
    if vocabulary is not None:
        x = encode_texts(texts_or_x, vocabulary, MAX_LENGTH)
    else:
        x = texts_or_x
    model.eval(); out = []
    with torch.no_grad():
        for part in batches(x, batch_size):
            tokens = torch.as_tensor(np.asarray(part), dtype=torch.long, device=device)
            lengths = torch.as_tensor(sequence_lengths(np.asarray(part)), dtype=torch.long, device=device)
            out.extend((torch.sigmoid(model(tokens, lengths)) >= 0.5).cpu().numpy().astype(int))
    return np.asarray(out)


def train_mlp(train_texts, train_y, val_texts, val_y, device):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30_000, min_df=2, max_df=0.95,
                                 sublinear_tf=True, preprocessor=clean_text, tokenizer=tokenize, token_pattern=None)
    train_x, val_x = vectorizer.fit_transform(train_texts), vectorizer.transform(val_texts)
    model = TfidfMLP(train_x.shape[1]).to(device)
    criterion, optimizer = nn.BCEWithLogitsLoss(), AdamW(model.parameters(), lr=1e-3)
    order, best, waiting, history = np.arange(train_x.shape[0]), -1.0, 0, []
    for epoch in range(1, 9):
        np.random.shuffle(order); model.train()
        for idx in batches(order, 128):
            features = torch.as_tensor(train_x[idx].toarray(), dtype=torch.float32, device=device)
            labels = torch.as_tensor(train_y[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(); loss = criterion(model(features), labels); loss.backward(); optimizer.step()
        predicted = predict_mlp(model, vectorizer, val_texts, device)
        score = metrics(val_y, predicted)["macro_f1"]; history.append({"epoch": epoch, "validation_macro_f1": score})
        print(f"mlp: epoch {epoch}, validation macro F1={score:.4f}", flush=True)
        if score > best:
            best, waiting = score, 0; torch.save(model.state_dict(), OUT / "mlp_best.pt")
        else:
            waiting += 1
            if waiting >= 2: break
    model.load_state_dict(torch.load(OUT / "mlp_best.pt", map_location=device, weights_only=True))
    import joblib
    joblib.dump(vectorizer, OUT / "mlp_tfidf.joblib")
    write_csv(OUT / "mlp_history.csv", history)
    return model, vectorizer


def predict_mlp(model, vectorizer, texts, device):
    model.eval(); predictions = []
    with torch.no_grad():
        for part in batches(texts, 256):
            x = torch.as_tensor(vectorizer.transform(part).toarray(), dtype=torch.float32, device=device)
            predictions.extend((torch.sigmoid(model(x)) >= 0.5).cpu().numpy().astype(int))
    return np.asarray(predictions)


def bert_loader(texts, labels, tokenizer, batch_size, shuffle=False):
    class BertSet(Dataset):
        def __len__(self): return len(texts)
        def __getitem__(self, i): return texts[i], int(labels[i])
    def collate(records):
        sentences, targets = zip(*records)
        inputs = tokenizer(list(sentences), padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs["labels"] = torch.tensor(targets)
        return inputs
    return DataLoader(BertSet(), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def train_bert(train_texts, train_y, val_texts, val_y, device):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    best, waiting, history = -1.0, 0, []
    for epoch in range(1, 4):
        model.train()
        for batch in bert_loader(train_texts, train_y, tokenizer, 16, shuffle=True):
            labels = batch.pop("labels").to(device); batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(); output = model(**batch, labels=labels); output.loss.backward(); optimizer.step()
        predicted = predict_bert(model, tokenizer, val_texts, device)
        score = metrics(val_y, predicted)["macro_f1"]; history.append({"epoch": epoch, "validation_macro_f1": score})
        print(f"distilbert: epoch {epoch}, validation macro F1={score:.4f}", flush=True)
        if score > best:
            best, waiting = score, 0; model.save_pretrained(OUT / "distilbert_best"); tokenizer.save_pretrained(OUT / "distilbert_best")
        else:
            waiting += 1
            if waiting >= 2: break
    model = AutoModelForSequenceClassification.from_pretrained(OUT / "distilbert_best").to(device)
    write_csv(OUT / "distilbert_history.csv", history)
    return model, tokenizer


def predict_bert(model, tokenizer, texts, device):
    model.eval(); predictions = []
    with torch.no_grad():
        for part in batches(texts, 32):
            batch = tokenizer(part, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            predictions.extend(torch.argmax(model(**batch).logits, dim=1).cpu().numpy())
    return np.asarray(predictions)


SYNONYMS = {"good": "nice", "great": "excellent", "bad": "poor", "boring": "dull", "love": "like",
            "hate": "dislike", "amazing": "wonderful", "awful": "terrible", "funny": "humorous", "slow": "sluggish"}


def character_noise(text, rng):
    tokens = text.split()
    candidates = [i for i, token in enumerate(tokens) if len(token) > 4]
    if candidates:
        i = rng.choice(candidates); token = tokens[i]; j = rng.randrange(len(token) - 1)
        tokens[i] = token[:j] + token[j + 1] + token[j] + token[j + 2:]
    return " ".join(tokens)


def synonym_substitution(text, rng):
    tokens = text.split()
    candidates = [i for i, token in enumerate(tokens) if token.lower().strip(".,!?'") in SYNONYMS]
    if candidates:
        i = rng.choice(candidates); core = tokens[i].lower().strip(".,!?'"); tokens[i] = SYNONYMS[core]
    return " ".join(tokens)


def random_deletion(text, rng):
    tokens = text.split()
    if len(tokens) > 3:
        del tokens[rng.randrange(len(tokens))]
    return " ".join(tokens)


def amazon_test(limit=2_000):
    """Balanced, bounded product-review sample so cross-domain evaluation is repeatable."""
    stream = load_dataset("amazon_polarity", split="test", streaming=True)
    texts, labels, quota = [], [], limit // 2
    counts = [0, 0]
    for item in stream:
        label = int(item["label"])
        if counts[label] < quota:
            texts.append(item["content"]); labels.append(label); counts[label] += 1
        if counts == [quota, quota]: break
    return texts, np.asarray(labels)


def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)


def write_evidence(name, texts, labels, predictions):
    rows = []
    for text, actual, predicted in zip(texts, labels, predictions):
        if actual != predicted and len(rows) < 20:
            rows.append({"review": text[:1000], "actual": int(actual), "predicted": int(predicted)})
    write_csv(OUT / f"{name}_error_samples.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    selected = MODELS if args.all or not args.model else (args.model,)
    OUT.mkdir(parents=True, exist_ok=True); set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}. Full suite is intended for a Colab/Kaggle GPU.", flush=True)
    if device.type != "cuda": raise RuntimeError("No CUDA GPU detected. Run this suite in a GPU Colab/Kaggle runtime.")
    train_texts, train_y, val_texts, val_y, test_texts, test_y = full_imdb()
    vocabulary = build_vocabulary(train_texts, VOCAB_SIZE)
    (OUT / "vocabulary.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    train_x, val_x, test_x = (encode_texts(items, vocabulary, MAX_LENGTH) for items in (train_texts, val_texts, test_texts))
    amazon_texts, amazon_y = amazon_test()
    rng = random.Random(SEED); robustness_texts, robustness_y = test_texts[:1_000], test_y[:1_000]
    attacks = {"character_noise": [character_noise(t, rng) for t in robustness_texts],
               "synonym_substitution": [synonym_substitution(t, rng) for t in robustness_texts],
               "random_word_deletion": [random_deletion(t, rng) for t in robustness_texts]}
    comparison = []
    for name in selected:
        started = time.time()
        if name == "mlp":
            model, vectorizer = train_mlp(train_texts, train_y, val_texts, val_y, device)
            predict = lambda items: predict_mlp(model, vectorizer, items, device)
        elif name == "textcnn":
            model = TextCNN(len(vocabulary)).to(device)
            model = train_token_model(model, train_x, train_y, val_x, val_y, device, name)
            predict = lambda items: predict_token(model, items, device, vocabulary=vocabulary)
        elif name == "bilstm":
            model = BiLSTMSentimentClassifier(len(vocabulary), 128, 128, 0.5).to(device)
            model = train_token_model(model, train_x, train_y, val_x, val_y, device, name)
            predict = lambda items: predict_token(model, items, device, vocabulary=vocabulary)
        else:
            model, tokenizer = train_bert(train_texts, train_y, val_texts, val_y, device)
            predict = lambda items: predict_bert(model, tokenizer, items, device)
        clean = predict(test_texts); clean_metrics = metrics(test_y, clean); write_evidence(name, test_texts, test_y, clean)
        row = {"model": name, **clean_metrics, "cross_domain_accuracy": metrics(amazon_y, predict(amazon_texts))["accuracy"],
               "runtime_minutes": round((time.time() - started) / 60, 2)}
        for attack, altered in attacks.items():
            row[f"{attack}_accuracy"] = metrics(robustness_y, predict(altered))["accuracy"]
            row[f"{attack}_drop"] = clean_metrics["accuracy"] - row[f"{attack}_accuracy"]
        comparison.append(row); print(json.dumps(row, indent=2), flush=True)
    write_csv(OUT / "final_comparison.csv", comparison)
    (OUT / "run_metadata.json").write_text(json.dumps({"seed": SEED, "device": str(device), "models": list(selected),
        "train": 20000, "validation": 5000, "test": 25000, "robustness_test_size": 1000, "cross_domain_test_size": 2000}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

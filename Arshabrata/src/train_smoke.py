import csv
import json
import sys
import time
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import experiment_config
from data import ReviewDataset, set_seed
from model import BiLSTMSentimentClassifier


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, labels, predictions = [], [], []
    with torch.no_grad():
        for tokens, lengths, targets in loader:
            tokens, lengths, targets = tokens.to(device), lengths.to(device), targets.to(device)
            logits = model(tokens, lengths)
            losses.append(criterion(logits, targets).item() * len(targets))
            labels.extend(targets.cpu().numpy().astype(int))
            predictions.extend((torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int))
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    return {
        "loss": sum(losses) / len(labels), "accuracy": accuracy_score(labels, predictions),
        "precision": precision, "recall": recall, "macro_f1": f1,
        "labels": labels, "predictions": predictions,
    }


def plot_history(history, output: Path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE loss")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in axes: axis.legend(); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Train the final 20k/5k/25k experiment.")
    args = parser.parse_args()
    config = experiment_config(full=args.full)
    tag = "full" if args.full else "smoke"
    for directory in config.output_paths.values(): directory.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    archive = config.output_paths["processed"] / f"{tag}_data_seed42.npz"
    if not archive.exists():
        raise FileNotFoundError("Run src/prepare_smoke_data.py (with --full for the final run) before training.")
    cached = np.load(archive)
    datasets = {
        "train": ReviewDataset(cached["train_x"], cached["train_lengths"], cached["train_y"]),
        "validation": ReviewDataset(cached["validation_x"], cached["validation_lengths"], cached["validation_y"]),
        "test": ReviewDataset(cached["test_x"], cached["test_lengths"], cached["test_y"]),
    }
    loaders = {name: DataLoader(ds, batch_size=config.batch_size, shuffle=(name == "train"), num_workers=0)
               for name, ds in datasets.items()}

    model = BiLSTMSentimentClassifier(int(cached["vocab_size"][0]), config.embedding_dim, config.hidden_dim, config.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_f1, stale_epochs, history = -1.0, 0, []
    checkpoint = config.output_paths["models"] / f"bilstm_{tag}_best.pt"

    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter(); model.train(); losses, labels, predictions = [], [], []
        for tokens, lengths, targets in loaders["train"]:
            tokens, lengths, targets = tokens.to(device), lengths.to(device), targets.to(device)
            optimizer.zero_grad(); logits = model(tokens, lengths); loss = criterion(logits, targets)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(loss.item() * len(targets)); labels.extend(targets.cpu().numpy().astype(int))
            predictions.extend((torch.sigmoid(logits) >= 0.5).detach().cpu().numpy().astype(int))
        validation = evaluate(model, loaders["validation"], criterion, device)
        row = {"epoch": epoch, "train_loss": sum(losses) / len(labels), "train_accuracy": accuracy_score(labels, predictions),
               "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
               "validation_macro_f1": validation["macro_f1"], "seconds": time.perf_counter() - started}
        history.append(row); print(row)
        if validation["macro_f1"] > best_f1:
            best_f1, stale_epochs = validation["macro_f1"], 0
            torch.save({"model_state": model.state_dict(), "vocab_size": int(cached["vocab_size"][0]), "config": config.__dict__}, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience: print("Early stopping."); break

    with (config.output_paths["results"] / f"{tag}_training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    plot_history(history, config.output_paths["results"] / f"{tag}_learning_curves.png")
    saved = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(saved["model_state"])
    test = evaluate(model, loaders["test"], criterion, device)
    report = {key: round(float(test[key]), 4) for key in ("loss", "accuracy", "precision", "recall", "macro_f1")}
    report.update({"dataset": f"IMDB {'full experiment' if args.full else 'smoke test'}", "train_reviews": config.train_size, "validation_reviews": config.validation_size,
                   "test_reviews": config.test_size, "seed": config.seed, "device": str(device), "best_validation_macro_f1": round(best_f1, 4)})
    (config.output_paths["results"] / f"{tag}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cm = confusion_matrix(test["labels"], test["predictions"])
    display = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]); display.plot(cmap="Blues", colorbar=False)
    plt.title(f"BiLSTM {tag}-test confusion matrix"); plt.tight_layout(); plt.savefig(config.output_paths["results"] / f"{tag}_confusion_matrix.png", dpi=180); plt.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

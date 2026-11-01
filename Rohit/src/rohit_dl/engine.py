import csv
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from .text import build_vocab, encode
from .model import TextCNN
from .io import load_json, save_json, read_jsonl, write_jsonl, sha256


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loader(rows, vocab, config, shuffle=False):
    x = torch.tensor([encode(r["text"], vocab, config["max_length"]) for r in rows], dtype=torch.long)
    y = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(TensorDataset(x, y), batch_size=config["batch_size"], shuffle=shuffle,
                      generator=generator, num_workers=0)


def metrics(labels, predictions):
    p, r, f, _ = precision_recall_fscore_support(labels, predictions, labels=[0, 1], average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
            "n": len(labels)}


@torch.no_grad()
def infer(model, batches, device):
    model.eval()
    probabilities, labels = [], []
    total_loss = 0.0
    for x, y in batches:
        logits = model(x.to(device))
        total_loss += torch.nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
        probabilities.extend(logits.softmax(1)[:, 1].cpu().tolist())
        labels.extend(y.tolist())
    predictions = [int(p >= 0.5) for p in probabilities]
    result = metrics(labels, predictions)
    result["loss"] = total_loss/len(labels)
    return result, probabilities


def make_model(vocab, config):
    return TextCNN(len(vocab), **{k: config[k] for k in ("embedding_dim", "num_filters", "kernel_sizes", "dropout")})


def train(data_dir, run_dir, config_path):
    config = load_json(config_path)
    if config["epochs"] < 1 or config["patience"] < 1 or config["batch_size"] < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    if run_dir.exists():
        raise FileExistsError("Run directory exists; choose a fresh run directory")
    train_rows, val_rows = read_jsonl(data_dir/"train.jsonl"), read_jsonl(data_dir/"validation.jsonl")
    if {r["id"] for r in train_rows} & {r["id"] for r in val_rows}:
        raise ValueError("Train/validation IDs overlap")
    set_seed(config["seed"])
    torch.set_num_threads(config.get("threads", 4))
    device = device_for(config["device"])
    vocab = build_vocab((r["text"] for r in train_rows), config["vocab_size"])
    train_loader, val_loader = loader(train_rows, vocab, config, True), loader(val_rows, vocab, config)
    model = make_model(vocab, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    run_dir.mkdir(parents=True)
    save_json(run_dir/"config.json", config)
    save_json(run_dir/"vocab.json", vocab)
    save_json(run_dir/"provenance.json", {"python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": str(device), "train_count": len(train_rows),
        "validation_count": len(val_rows), "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_sha256": sha256(data_dir/"train.jsonl"), "validation_sha256": sha256(data_dir/"validation.jsonl"),
        "config_sha256": sha256(config_path), "selection_rule": "lowest validation cross entropy",
        "determinism": "Fixed seeds; device kernels can still differ; deterministic algorithms warn_only=True"})
    # Keep development IDs to reject accidental train/validation evaluation as held-out results.
    save_json(run_dir/"development_ids.json", [r["id"] for r in train_rows+val_rows])
    history, best, stale = [], float("inf"), 0
    start = time.monotonic()
    for epoch in range(1, config["epochs"]+1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = torch.nn.functional.cross_entropy(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item()*len(y)
        val, _ = infer(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running/len(train_rows), "validation_loss": val["loss"],
               "validation_accuracy": val["accuracy"], "validation_f1_macro": val["f1_macro"],
               "elapsed_seconds": time.monotonic()-start}
        history.append(row)
        save_json(run_dir/"history.json", history)
        print(row, flush=True)
        if val["loss"] < best:
            best, stale = val["loss"], 0
            torch.save(model.state_dict(), run_dir/"best.pt")
            save_json(run_dir/"best_validation.json", {"epoch": epoch, **val})
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    save_json(run_dir/"training_complete.json", {"epochs_run": len(history), "elapsed_seconds": time.monotonic()-start})
    return run_dir


def load_run(run_dir, device_name="auto"):
    run_dir = Path(run_dir)
    if not (run_dir/"training_complete.json").exists():
        raise ValueError("Training did not complete; no finalized checkpoint available")
    config, vocab = load_json(run_dir/"config.json"), load_json(run_dir/"vocab.json")
    torch.set_num_threads(config.get("threads", 4))
    device = device_for(device_name)
    model = make_model(vocab, config)
    model.load_state_dict(torch.load(run_dir/"best.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, vocab, config, device


def evaluate(run_dir, input_path, output_dir, name="clean", device="auto"):
    rows = read_jsonl(input_path)
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Evaluation output exists; use a fresh output directory")
    dev_ids = set(load_json(run_dir/"development_ids.json"))
    if any(r["id"] in dev_ids for r in rows):
        raise ValueError("Evaluation IDs overlap training or validation")
    model, vocab, config, device = load_run(run_dir, device)
    result, probs = infer(model, loader(rows, vocab, config), device)
    result.update({"name": name, "input_sha256": sha256(input_path), "checkpoint_sha256": sha256(run_dir/"best.pt")})
    output_dir.mkdir(parents=True)
    save_json(output_dir/"metrics.json", result)
    predictions = [{"id": r["id"], "text": r["text"], "label": r["label"], "prediction": int(p >= .5),
                    "probability_positive": p} for r, p in zip(rows, probs)]
    write_jsonl(output_dir/"predictions.jsonl", predictions)
    errors = sorted((r for r in predictions if r["label"] != r["prediction"]),
                    key=lambda r: abs(r["probability_positive"]-.5), reverse=True)
    write_jsonl(output_dir/"errors.jsonl", errors)
    plot_confusion(result["confusion_matrix"], output_dir/"confusion_matrix.png", name)
    print(result, flush=True)
    return result, predictions


def plot_confusion(matrix, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black",
                    bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
           yticklabels=["Negative", "Positive"], xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def predict(run_dir, text, device="auto"):
    model, vocab, config, device = load_run(run_dir, device)
    with torch.no_grad():
        p = model(torch.tensor([encode(text, vocab, config["max_length"])], device=device)).softmax(1)[0, 1].item()
    return {"label": "positive" if p >= .5 else "negative", "probability_positive": p}

import argparse
import csv
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .common import ROOT, digest, metrics, read_json, write_json


def predict(model, tokenizer, data, max_length, batch_size):
    predictions, probabilities = [], []
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=model.device.type,
                                                enabled=model.device.type == "cuda"):
        for start in range(0, len(data), batch_size):
            batch = tokenizer(data[start:start + batch_size]["text"], padding=True,
                              truncation=True, max_length=max_length, return_tensors="pt")
            logits = model(**{k: v.to(model.device) for k, v in batch.items()}).logits
            scores = logits.softmax(-1).cpu().numpy()
            predictions.extend(scores.argmax(-1).tolist())
            probabilities.extend(scores[:, 1].tolist())
    return predictions, probabilities


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models/distilbert"))
    p.add_argument("--clean-only", action="store_true")
    args = p.parse_args()
    experiment = read_json(f"{args.model}/experiment.json")
    if experiment["smoke_only"]:
        raise ValueError("Smoke checkpoints cannot produce final project results")
    manifest = read_json(ROOT / "data/manifest.json")
    if manifest != experiment["data_manifest"]:
        raise ValueError("Prepared data differs from this checkpoint's training manifest")
    cfg = experiment["config"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    data = load_from_disk(str(ROOT / "data/prepared"))
    conditions = {"clean": data["test"], "amazon": data["amazon"]}
    perturbation_manifest = None
    if not args.clean_only:
        perturbation_manifest = read_json(ROOT / "data/perturbed/manifest.json")
        for kind in ["synonym", "typo", "deletion"]:
            for rate in cfg["perturbation_rates"]:
                key = f"{kind}_{rate:g}"
                conditions[key] = load_from_disk(str(ROOT / "data/perturbed" / key))
                if conditions[key]._fingerprint != perturbation_manifest["conditions"][key]["fingerprint"]:
                    raise ValueError(f"Perturbation fingerprint mismatch: {key}")
    out = ROOT / "results" / ("clean_only" if args.clean_only else "full")
    if out.exists():
        raise FileExistsError(f"{out} already exists; preserve earlier results")
    out.mkdir(parents=True)
    results = []
    clean_preds = None
    for name, ds in conditions.items():
        labels = np.asarray(ds["label"])
        preds, probs = predict(model, tokenizer, ds, cfg["max_length"], cfg["eval_batch_size"])
        row = {"model": "DistilBERT", "condition": name, **metrics(labels, preds)}
        if name == "clean":
            clean_preds = np.asarray(preds)
            clean_acc = row["accuracy"]
        row["accuracy_drop_pp"] = 100 * (clean_acc - row["accuracy"])
        if name not in {"clean", "amazon"}:
            if list(ds["id"]) != list(data["test"]["id"]) or list(ds["label"]) != list(data["test"]["label"]):
                raise ValueError("Perturbed examples must align with clean IDs and labels")
            correct = clean_preds == labels
            row["prediction_flip_rate"] = float(np.mean(clean_preds != preds))
            row["failure_rate_on_clean_correct"] = float(np.mean(np.asarray(preds)[correct] != labels[correct])) if correct.any() else None
            row["changed_review_fraction"] = float(np.mean(ds["changed"]))
        write_json(out / f"{name}.json", row)
        with (out / f"{name}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "prediction", "positive_probability"])
            writer.writerows(zip(ds["id"], labels.tolist(), preds, probs))
        # Inspectable errors without duplicating the entire review corpus in Git.
        error_ids = [int(i) for i in np.argsort(-np.abs(np.asarray(probs) - 0.5))
                     if preds[i] != labels[i]][:30]
        errors = [{"id": ds[i]["id"], "label": int(labels[i]), "prediction": preds[i],
                   "positive_probability": probs[i], "text": ds[i]["text"]} for i in error_ids]
        write_json(out / f"{name}_errors.json", errors)
        results.append(row)
        print(name, row, flush=True)
    fields = ["model", "condition", "n", "accuracy", "precision_macro", "recall_macro", "f1_macro", "accuracy_drop_pp"]
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    write_json(out / "provenance.json", {"experiment": experiment, "data_manifest": manifest,
              "experiment_hash": digest(experiment), "conditions": list(conditions),
              "perturbation_manifest": perturbation_manifest,
              "inference_precision": "CUDA float16 autocast" if model.device.type == "cuda" else "CPU float32"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([r["condition"] for r in results], [100 * r["accuracy"] for r in results])
    ax.set(ylabel="Accuracy (%)", ylim=(0, 100), title="DistilBERT sentiment robustness")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

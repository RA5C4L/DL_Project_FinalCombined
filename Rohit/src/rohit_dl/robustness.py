import csv
from pathlib import Path
from .engine import evaluate
from .io import load_json, read_jsonl, save_json, sha256


def run_suite(run_dir, clean_path, perturb_dir, output_dir, ood_path=None, device="auto"):
    perturb_dir, output_dir = Path(perturb_dir), Path(output_dir)
    manifest = load_json(perturb_dir/"manifest.json")
    if manifest["source_sha256"] != sha256(clean_path):
        raise ValueError("Perturbation source does not match clean data")
    clean_rows = read_jsonl(clean_path)
    signature = [(r["id"], r["label"]) for r in clean_rows]
    for item in manifest["sets"]:
        path = perturb_dir/item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Perturbation file checksum mismatch")
        rows = read_jsonl(path)
        if [(r["id"], r["label"]) for r in rows] != signature:
            raise ValueError("Perturbed IDs/labels/order must match clean data")
    if output_dir.exists():
        raise FileExistsError("Suite output exists; choose a fresh directory")
    clean, clean_predictions = evaluate(run_dir, clean_path, output_dir/"clean", "IMDB clean", device)
    clean_correct = {r["id"] for r in clean_predictions if r["prediction"] == r["label"]}
    summary = []
    def add(result, kind, rate=None, paired_failure=None, realized_rate=None, unchanged_count=None):
        summary.append({"condition": kind, "requested_rate": rate, "mean_realized_rate": realized_rate,
            "unchanged_count": unchanged_count, "n": result["n"],
            **{k: result[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
            "accuracy_drop_pp": 100*(clean["accuracy"]-result["accuracy"]),
            "clean_correct_failure_rate": paired_failure})
    add(clean, "clean")
    if ood_path:
        ood, _ = evaluate(run_dir, ood_path, output_dir/"ood", "OOD", device)
        add(ood, "ood")
    for item in manifest["sets"]:
        name = item["name"]
        result, predictions = evaluate(run_dir, perturb_dir/item["path"], output_dir/name, name, device)
        failed = sum(r["id"] in clean_correct and r["label"] != r["prediction"] for r in predictions)
        kind, rate = name.rsplit("_", 1)
        add(result, kind, float(rate), failed/len(clean_correct) if clean_correct else None,
            item.get("mean_realized_rate"), item.get("unchanged_count"))
    save_json(output_dir/"summary.json", summary)
    with (output_dir/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in ("synonym", "typo", "deletion"):
        points = [r for r in summary if r["condition"] == kind]
        ax.plot([0]+[100*r["requested_rate"] for r in points],
                [100*clean["accuracy"]]+[100*r["accuracy"] for r in points], marker="o", label=kind)
    ax.set(xlabel="Requested word perturbation rate (%)", ylabel="Accuracy (%)", title="TextCNN corruption robustness")
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir/"robustness.png", dpi=160); plt.close(fig)
    return summary

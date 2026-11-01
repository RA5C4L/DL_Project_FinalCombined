"""Generate shareable evidence from measured runs, without copying raw reviews."""
import csv
import shutil
from pathlib import Path
from .io import load_json, save_json, read_jsonl, sha256
from .text import clean_text, tokenize


def export_report(run_dir, suite_dir, data_dir, output_dir):
    run_dir, suite_dir, data_dir, output_dir = map(Path, (run_dir, suite_dir, data_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("Report export exists; use a fresh directory")
    complete = load_json(run_dir / "training_complete.json")
    config = load_json(run_dir / "config.json")
    provenance = load_json(run_dir / "provenance.json")
    history = load_json(run_dir / "history.json")
    best = load_json(run_dir / "best_validation.json")
    summary = load_json(suite_dir / "summary.json")
    vocab = load_json(run_dir / "vocab.json")
    checkpoint_hash = sha256(run_dir / "best.pt")
    clean = load_json(suite_dir / "clean/metrics.json")
    if clean["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Report run and evaluation checkpoint differ")
    output_dir.mkdir(parents=True)
    for filename in ("config.json", "provenance.json", "history.json", "best_validation.json", "training_complete.json"):
        shutil.copy2(run_dir/filename, output_dir/filename)
    for filename in ("summary.json", "summary.csv", "robustness.png"):
        shutil.copy2(suite_dir/filename, output_dir/filename)
    shutil.copy2(suite_dir/"clean/confusion_matrix.png", output_dir/"clean_confusion_matrix.png")
    if (suite_dir/"ood/confusion_matrix.png").exists():
        shutil.copy2(suite_dir/"ood/confusion_matrix.png", output_dir/"ood_confusion_matrix.png")
    if (data_dir/"manifest.json").exists():
        shutil.copy2(data_dir/"manifest.json", output_dir/"data_manifest.json")
    audit = {}
    hashes_by_split = {}
    for name in ("train", "validation", "test", "ood"):
        path = data_dir/f"{name}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        tokens = [tokenize(r["text"]) for r in rows]
        count = sum(len(t) for t in tokens)
        hashes_by_split[name] = {__import__('hashlib').sha256(clean_text(r["text"]).lower().encode()).hexdigest() for r in rows}
        audit[name] = {"n": len(rows), "negative": sum(r["label"] == 0 for r in rows),
            "positive": sum(r["label"] == 1 for r in rows),
            "truncated_count": sum(len(t) > config["max_length"] for t in tokens),
            "empty_count": sum(not t for t in tokens),
            "oov_fraction_before_truncation": sum(w not in vocab for t in tokens for w in t)/max(count, 1),
            "duplicate_cleaned_text_count": len(rows)-len(hashes_by_split[name])}
        del tokens, rows
    audit["exact_cleaned_text_overlap"] = {f"{a}_vs_{b}": len(hashes_by_split[a] & hashes_by_split[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    from .engine import metrics
    development_hashes = hashes_by_split["train"] | hashes_by_split["validation"]
    test_predictions = read_jsonl(suite_dir/"clean/predictions.jsonl")
    retained = [r for r in test_predictions if __import__('hashlib').sha256(clean_text(r["text"]).lower().encode()).hexdigest() not in development_hashes]
    audit["test_excluding_development_text_matches"] = {
        "excluded_count": len(test_predictions)-len(retained),
        "purpose": "Descriptive sensitivity check only; official full test score remains primary",
        **(metrics([r["label"] for r in retained], [r["prediction"] for r in retained]) if retained else {"n": 0, "accuracy": None})}
    save_json(output_dir/"data_audit.json", audit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], marker='o', label='Train')
    axes[0].plot(epochs, [r["validation_loss"] for r in history], marker='o', label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy', title='Checkpoint selection')
    axes[0].axvline(best["epoch"], linestyle='--', color='gray', label='Selected epoch')
    axes[0].legend()
    axes[1].plot(epochs, [100*r["validation_accuracy"] for r in history], marker='o')
    axes[1].set(xlabel='Epoch', ylabel='Validation accuracy (%)', title='Development performance')
    fig.tight_layout(); fig.savefig(output_dir/'training_curves.png', dpi=160); plt.close(fig)
    sensitivity = audit["test_excluding_development_text_matches"]
    sensitivity_text = (
        f"After excluding {sensitivity['excluded_count']} test rows whose cleaned text matches training or validation, "
        f"the same checkpoint scores {100*sensitivity['accuracy']:.2f}% accuracy on {sensitivity['n']:,} retained rows. "
        "This sensitivity check does not replace the official test result."
    ) if sensitivity["n"] else "Every test text matches development data; no non-overlapping sensitivity score can be computed."
    lines = ["# Measured TextCNN results", "", "Rohit Mohanty — ICT 4442", "",
        f"Training used {provenance['train_count']:,} IMDB reviews and {provenance['validation_count']:,} validation reviews. "
        f"The selected checkpoint is epoch {best['epoch']} of {complete['epochs_run']} completed epochs, selected by minimum validation loss. "
        f"Device: {provenance['device']}. Training seed: {config['seed']}. Parameters: {provenance['parameter_count']:,}.", "",
        "These are measured single-run results. The interim report projections are not used as evidence.", "",
        "| Condition | Requested rate | Realized rate | N | Accuracy % | Macro precision % | Macro recall % | Macro F1 % | Drop pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        rate = '—' if r['requested_rate'] is None else f"{100*r['requested_rate']:g}%"
        realized = "—" if r.get("mean_realized_rate") is None else f"{100*r['mean_realized_rate']:.2f}%"
        lines.append(f"| {r['condition']} | {rate} | {realized} | {r['n']:,} | {100*r['accuracy']:.2f} | {100*r['precision_macro']:.2f} | {100*r['recall_macro']:.2f} | {100*r['f1_macro']:.2f} | {r['accuracy_drop_pp']:.2f} |")
    lines += ["", "## Interpretation limits", "",
        "The corruption tests are seeded, model-independent edits. Their inherited labels assume unchanged sentiment, which is not guaranteed. Human semantic review is still needed before treating the perturbation rows as label-preserving adversarial robustness results. Clean and OOD evaluations use their benchmark labels.", "",
        "The model uses the first 400 tokens. Some generated edits lie beyond that window. This is one seed and one architecture; no superiority claim over the teammates' models follows until their measured results use the same protocol.", "",
        "The data audit reports existing exact-text duplicates and overlaps without changing the specified official split. These can make evaluation optimistic. Any deduplicated sensitivity analysis must be a separate team-agreed experiment, not an undisclosed change to these benchmark results.", "",
        sensitivity_text, "",
        "## Reproducibility", "", f"Checkpoint SHA-256: `{checkpoint_hash}`.", "",
        "See configuration, provenance, dataset manifest, training history, data audit and the full-precision summary CSV in this folder. The raw predictions and model checkpoint remain under the local run directory and are ignored by Git.", "",
        "![Training curves](training_curves.png)", "", "![Clean confusion matrix](clean_confusion_matrix.png)", "",
        "![Robustness curve](robustness.png)", ""]
    (output_dir/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"Exported measured evidence to {output_dir}", flush=True)

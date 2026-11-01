"""Generate a report section from measured outputs, without invented results."""
import csv
from .common import ROOT, read_json


def main():
    folder = ROOT / "results/full"
    with (folder / "comparison.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    provenance = read_json(folder / "provenance.json")
    experiment = provenance["experiment"]
    expected = {"clean", "amazon"} | {
        f"{kind}_{rate:g}" for kind in ["synonym", "typo", "deletion"]
        for rate in experiment["config"]["perturbation_rates"]}
    if {r["condition"] for r in rows} != expected:
        raise ValueError("A complete report requires clean, cross-domain and all perturbation results")
    clean = next(r for r in rows if r["condition"] == "clean")
    amazon = next(r for r in rows if r["condition"] == "amazon")
    worst = max((r for r in rows if r["condition"] not in {"clean", "amazon"}),
                key=lambda r: float(r["accuracy_drop_pp"]))
    lines = ["# DistilBERT implementation and measured evaluation", "",
        "Project owner: Navneet Vivek (230953352), ICT 4442.", "",
        f"The fine-tuned DistilBERT achieved {100*float(clean['accuracy']):.2f}% accuracy on the complete 25,000-review IMDB test set and "
        f"{100*float(amazon['accuracy']):.2f}% on the fixed 25,000-review Amazon test sample. "
        f"The largest observed perturbation-related accuracy drop was {float(worst['accuracy_drop_pp']):.2f} percentage points for {worst['condition']}.", "",
        "## Implementation", "",
        "The generic pretrained uncased DistilBERT encoder was fine-tuned with a binary classification head. "
        "The official IMDB training split was divided into 20,000 training and 5,000 validation examples using stratified sampling with seed 42. "
        "The input limit was 256 WordPiece tokens. AdamW used a learning rate of 2e-5, weight decay 0.01, 10% warmup, two epochs and effective batch size 16. "
        "The checkpoint with the best validation macro F1 was selected before final test evaluation.", "",
        "## Measured results", "",
        "Accuracy, precision, recall and F1 below are percentages. Precision, recall and F1 are macro averages. Drops are percentage points relative to clean IMDB.", "",
        "| Condition | N | Accuracy | Precision | Recall | Macro F1 | Accuracy drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        values = [100 * float(row[k]) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]]
        lines.append(f"| {row['condition']} | {row['n']} | " + " | ".join(f"{x:.2f}" for x in values)
                     + f" | {float(row['accuracy_drop_pp']):.2f} |")
    lines += ["", "## Error analysis", "",
        "The clean-test confusion matrix is shown below. Rows are true labels and columns are predictions.", "",
        "| True label | Predicted negative | Predicted positive |", "| --- | ---: | ---: |"]
    matrix = read_json(folder / "clean.json")["confusion_matrix"]
    for label, values in zip(["negative", "positive"], matrix):
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += ["", "For each condition, the accompanying error JSON contains up to 30 confident errors for qualitative inspection. "
        "Prediction CSVs preserve review IDs, labels and positive-class probabilities. "
        "Perturbed-condition JSONs additionally give prediction flip rate and failure rate among clean-correct examples.", "",
        "## Interpretation and limitations", "",
        "These are measured results from one seeded run, not estimates from the interim report. "
        "The Amazon result is on a fixed stratified subset of its official test split; no Amazon training data was used. "
        "Its comparison with IMDB also reflects differences in review difficulty and length, so it does not isolate a causal effect of domain shift. "
        "Random synonym, typo and deletion stress tests were applied at 5%, 10% and 20% rates. "
        "Synonym candidates came from nlpaug's WordNet model. These are not optimized TextFooler attacks, and the edits are not guaranteed to preserve meaning or sentiment.", "",
        "Human semantic review remains pending in `data/perturbed/manual_review.csv`. "
        "Until that review is completed, use the term random perturbation robustness rather than certified meaning-preserving adversarial robustness. "
        "Truncation can hide later review content and edits. A single model and single seed cannot establish an architecture-level ranking or statistical significance. "
        "The group comparison still requires the other members' measured outputs on these same frozen tests.", "",
        "## Reproducibility", "",
        f"Split SHA-256: `{provenance['data_manifest']['split_hash']}`.", "",
        f"Inference mode: {provenance['inference_precision']}. Training hardware: {experiment['gpu']}.", "",
        "See `provenance.json`, `data/manifest.json`, `data/perturbation_manifest.json`, the saved tokenizer/checkpoint, and the environment snapshot for run details."]
    stats_path = ROOT / "results/data_statistics.json"
    if stats_path.exists():
        stats = read_json(stats_path)
        lines += ["", "## Input length analysis", "",
            f"The 256-token limit truncates {100*stats['test']['fraction_truncated_at_256']:.2f}% of clean IMDB test reviews "
            f"and {100*stats['amazon']['fraction_truncated_at_256']:.2f}% of the Amazon sample. "
            "This is a measured difference between the inputs, not proof that truncation caused the cross-domain score difference. "
            "See `docs/ERROR_ANALYSIS.md` for concrete mixed-sentiment and apparent label-inconsistency examples."]
    (folder / "REPORT_SECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(folder / "REPORT_SECTION.md")


if __name__ == "__main__":
    main()

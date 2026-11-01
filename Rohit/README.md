# IMDB TextCNN sentiment classifier

This is **Rohit Mohanty's (230953168)** standalone, reproducible TextCNN contribution to the **ICT 4442** project, *Robustness of Deep Learning Architectures for Sentiment Classification Under Distribution Shift and Adversarial Text Perturbation*. My part covers tokenization and padding, TextCNN implementation, training, clean and cross-domain evaluation, and text-perturbation robustness tests.

The full **20,000 training / 5,000 validation / 25,000 IMDB test** experiment is complete. The selected model achieved **87.08% IMDB accuracy** and **79.96% Yelp accuracy**. A separate small synthetic smoke test is included to check the pipeline before a full run.

## Architecture

`token IDs → 128-d embedding → parallel Conv1D (128 filters each, widths 3/4/5) → ReLU → masked global max pooling → concatenate → dropout 0.5 → linear classifier`

The vocabulary contains at most **30,000 entries**, learned from training reviews only. Inputs are padded or truncated to **400 tokens**. The model uses trainable random embeddings, two output logits, and cross-entropy loss. Padding-only convolution windows are excluded from max pooling.

## Run the smoke test

Run all commands from this folder. For a new environment on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
rohit-dl smoke --output runs/smoke
```

On macOS/Linux, create and activate the environment with `python3 -m venv .venv` and `source .venv/bin/activate`, then use the same installation and smoke commands.

The smoke test creates **64 training, 16 validation, and 16 test examples** from repeated synthetic sentences. It uses a reduced model to check training, checkpoint loading, metrics, prediction and plotting. It does **not** download IMDB or provide benchmark results.

Run `python -m pytest -q` for the automated checks. **14 tests passed** in the recorded local environment.

## Final experiment

For a fresh checkout, prepare the required full dataset and train TextCNN:

```bash
rohit-dl prepare --output data --ood yelp
rohit-dl train --data data --run runs/textcnn_seed42 --config configs/baseline.json
rohit-dl evaluate --run runs/textcnn_seed42 --input data/test.jsonl --output runs/clean_seed42
```

Preparation uses stratified sampling with **seed 42** and saves the split indices. Training uses Adam, a batch size of 64, and early stopping with patience 3. The checkpoint is selected by **minimum validation loss**, without using test results. The completed CPU run stopped after nine epochs and selected epoch six.

The complete local folder already contains the prepared data and trained run. Skip completed steps when reusing it; choose fresh output paths for new experiments. Existing data and run directories are protected from overwrite.

## Complete TextCNN robustness suite

The suite evaluates the same checkpoint on clean IMDB, all **38,000 Yelp test reviews**, and synonym substitution, character typos and word deletion at requested rates of **5%, 10% and 20%**.

Download the language resources:

```bash
python -m nltk.downloader -d .cache/nltk_data wordnet averaged_perceptron_tagger_eng
```

Set their location in the current shell:

```powershell
# Windows PowerShell
$env:NLTK_DATA="$PWD/.cache/nltk_data"
```

```bash
# macOS / Linux
export NLTK_DATA="$PWD/.cache/nltk_data"
```

Then generate the perturbed data and evaluate:

```bash
rohit-dl perturb --input data/test.jsonl --output data/perturbed --rates 0.05 0.10 0.20
rohit-dl robustness --run runs/textcnn_seed42 --clean data/test.jsonl --perturbed data/perturbed --ood data/ood.jsonl --output runs/robustness_seed42
rohit-dl report --run runs/textcnn_seed42 --suite runs/robustness_seed42 --data data --output reports/reproduced_seed42
```

Reuse existing perturbation files when working with the completed local folder. The report command uses a new output name to preserve the committed results. Predictions, errors, confusion matrices and full-precision metrics are saved under the evaluation run directory.

For a notebook workflow, use [TextCNN_Walkthrough.ipynb](notebooks/TextCNN_Walkthrough.ipynb) with the complete project folder available in the notebook environment. Automatic device selection prefers CUDA, then Apple MPS, then CPU.

## Measured results

| Evaluation | Accuracy | Macro F1 | Drop from clean |
|---|---:|---:|---:|
| IMDB clean | 87.08% | 87.05% | — |
| Yelp cross-domain | 79.96% | 79.85% | 7.12 pp |
| Synonyms, requested 20% | 85.44% | 85.42% | 1.64 pp |
| Typos, requested 20% | 82.60% | 82.59% | 4.48 pp |
| Deletion, requested 20% | 82.65% | 82.58% | 4.42 pp |

All **11 conditions and 288,000 predictions** were checked against saved metrics and the same checkpoint. Drops use unrounded scores; `pp` means percentage points.

Synonym edits reach only **4.37% actual word replacement** in the requested 20% condition because eligible candidates are limited. Typos and deletion reach about **19.72%**. These rows are therefore not equal-intensity comparisons.

See [all measured results and plots](reports/textcnn_seed42/RESULTS.md), [error analysis](reports/textcnn_seed42/ERROR_ANALYSIS.md), and [the full-precision CSV](reports/textcnn_seed42/summary.csv).

## Team contribution

The project assigns:

- **Varaang Tiwari:** MLP with TF-IDF.
- **Rohit Mohanty:** TextCNN, tokenizer/padding and evaluation — implemented here.
- **Arshabrata Bhaumik:** BiLSTM.
- **Navneet Vivek:** DistilBERT fine-tuning.
- **All members:** cross-domain and perturbation evaluation, error analysis and final comparison.

This repository contains the **TextCNN contribution**, not a runner for all four models. Share the saved split indices and exact evaluation datasets with teammates, then merge their measured results using the same metric definitions. See [my contribution notes](reports/ROHIT_CONTRIBUTION.md) and [the team handoff](docs/TEAM_HANDOFF.md).

## Integrity note

The synthetic smoke test is diagnostic; its scores must not be presented as final results. The interim report's **89.3% accuracy and 89.4% macro F1 were projections**, replaced here by actual measurements.

These are single-seed, model-independent corruption tests, not TextFooler or HotFlip attacks. Perturbations can change meaning; an AI-assisted nine-example spot check is documented in [the semantic review](reports/SEMANTIC_REVIEW.md). Existing IMDB text overlaps, truncation, and vocabulary limitations are disclosed in the reports. Do not alter measured values or infer superiority over teammates' models without a shared comparison.

## GitHub files and trained model

Source, tests, configuration, notebook, guides and small measured reports are ready for GitHub. Raw datasets, caches, original Word documents and trained run files are excluded by `.gitignore`. The GitHub-ready ZIP therefore does not contain trained weights.

The complete local model is in `runs/textcnn_seed42/`. After installing the package, use it with:

```bash
rohit-dl predict --run runs/textcnn_seed42 --text "The acting was excellent, but the story was not convincing."
```

To share the trained model separately, attach the **complete run folder** to an agreed release or model artifact location. A fresh checkout must train the model or obtain that folder before inference.

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for the comprehensive methodology, reproducibility records, references, acknowledgments and troubleshooting, and [GITHUB_GUIDE.md](docs/GITHUB_GUIDE.md) for upload instructions. The implementation and documentation were prepared with AI assistance; follow the course's acknowledgment requirements.

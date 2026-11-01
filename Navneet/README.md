# Navneet Vivek's DistilBERT project

Owner: **Navneet Vivek, 230953352** · ICT 4442

Your assigned contribution is fine-tuned DistilBERT implementation and evaluation for **Robustness of Deep Learning Architectures for Sentiment Classification Under Distribution Shift and Adversarial Text Perturbation**. The interim report additionally assigns you DistilBERT setup, robustness-library review and the perturbation plan. This folder contains that implementation from scratch.

Start with `docs/YOUR_PART.md` for responsibilities and `docs/EXPERIMENT_PROTOCOL.md` for the methodology. Read `STATUS.md` for what has actually been executed. Code being implemented is not evidence of trained-model performance.

For the complete starting point, development history, resolved issues, results and remaining work, read [PROJECT_DEVELOPMENT_REPORT.md](PROJECT_DEVELOPMENT_REPORT.md).

## Run on Windows

Open PowerShell in this folder. A local `.venv` may already be present on the original laptop; it is not portable and must not be committed. Otherwise install Python 3.12 and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m nltk.downloader -d .cache/nltk_data wordnet omw-1.4
$env:NLTK_DATA = "$PWD/.cache/nltk_data"
$env:HF_HOME = "$PWD/.cache/huggingface"
.\.venv\Scripts\python.exe -m pytest -q
```

CUDA is appropriate for this laptop's NVIDIA GPU. For CPU-only machines install `torch==2.8.0` from `https://download.pytorch.org/whl/cpu` instead. Training on CPU is substantially slower. Downloads include several GB of runtime packages, datasets and pretrained weights.

Then execute the stages in order:

```powershell
.\.venv\Scripts\python.exe -m navneet_dl.prepare
.\.venv\Scripts\python.exe -m navneet_dl.train --smoke
.\.venv\Scripts\python.exe -m navneet_dl.train
.\.venv\Scripts\python.exe -m navneet_dl.data_stats
.\.venv\Scripts\python.exe -m navneet_dl.perturb
.\.venv\Scripts\python.exe -m navneet_dl.verify
.\.venv\Scripts\python.exe -m navneet_dl.evaluate
.\.venv\Scripts\python.exe -m navneet_dl.report
.\.venv\Scripts\python.exe -m navneet_dl.predict "The acting was excellent and I enjoyed the movie."
```

`run_pipeline.ps1` executes prepare, production training, perturbations and evaluation with failure checks, skipping completed data/model stages. It will refuse to overwrite completed evaluation. The smoke model is separate and never accepted by the final evaluator. Training checkpoints support `python -m navneet_dl.train --resume runs/distilbert/checkpoint-N`.

**Team split:** If your group already has the common split, use `python -m navneet_dl.prepare --split-file path/to/team_splits.json` before preparing data. Its keys must be `train` and `validation`, with zero-based row indices into Hugging Face's official IMDB training split. Otherwise share the generated `data/splits.json` with all members. Seed 42 alone does not ensure identical splits across different algorithms or dataset orderings.

## Folder contents

| Path | Purpose |
| --- | --- |
| `navneet_dl/` | Dataset preparation, fine-tuning, perturbations, evaluation and demo prediction |
| `config.json` | Fixed model and experimental settings |
| `tests/` | Protocol, cleaning, metrics and perturbation checks |
| `notebooks/Run_in_Colab.ipynb` | Alternate GPU execution with the same source files |
| `docs/` | Role mapping, methodology, robustness review, viva and Git handoff |
| `references/` | Local copies of supplied documents, excluded from Git |
| `data/` | Shared indices and provenance; generated review datasets are ignored |
| `models/` | Saved tokenizer and trained checkpoint, ignored |
| `runs/` | Training checkpoints, logs and environment capture, ignored |
| `results/` | Genuine measured metrics, predictions, error examples and chart when evaluation finishes |

The model uses the pretrained WordPiece vocabulary. The report's 30,000-word learned vocabulary and 400-token sequences apply to the CNN/BiLSTM; its DistilBERT-specific setting is 256 tokens. All 20,000 training reviews are retained.

## Results and reporting

`results/full/comparison.csv` contains accuracy, macro precision/recall/F1 and accuracy drop in **percentage points**. Each condition also saves a confusion matrix (rows true, columns predicted; negative then positive), predictions by stable review ID, and up to 30 confident errors. The evaluation chart and provenance accompany the table.

`results/full/REPORT_SECTION.md` is generated directly from measured results for integration into the group report. `requirements-lock-windows.txt` records the exact tested Windows environment; the shorter `requirements.txt` is the starting point for a fresh platform installation.

Review `data/perturbed/manual_review.csv` before describing edits as meaning-preserving. The code uses random stress perturbations, not optimized TextFooler attacks. Synonyms may be contextually wrong and deletion can remove sentiment words. Rates are 5%, 10%, and 20% of alphabetic words, rounded down and capped by eligible edits; actual rates and unchanged examples are recorded.

Do not copy the interim report's projected MLP/TextCNN numbers into measured results. Do not claim that DistilBERT outperforms teammates until their real outputs use the same splits and perturbations.

## Primary technical references

- [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased)
- [Hugging Face text classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- [Amazon Polarity dataset](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
- [PyTorch installation](https://pytorch.org/get-started/locally/)

No Git repository is initialized and nothing is pushed by these scripts.

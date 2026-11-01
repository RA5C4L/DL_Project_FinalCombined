# IMDB BiLSTM sentiment classifier

This is Arshabrata Bhaumik's standalone, reproducible BiLSTM baseline for the ICT 4442 project. The first experiment is deliberately a smoke test: 256 training, 64 validation, and 128 untouched official-test reviews. It confirms that every pipeline component works before the 20,000/5,000/25,000 final experiment.

## Architecture

`token IDs -> 128-d embedding -> bidirectional LSTM (128 hidden units) -> dropout 0.5 -> linear classifier`

## Run the smoke test

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\train_smoke.py
```

Run `python .\src\prepare_smoke_data.py` first. It downloads IMDB, creates reproducible stratified subsets with seed 42, and saves split IDs, vocabulary, and encoded sequences. Then `python .\src\train_smoke.py` trains with early stopping and writes a checkpoint, metrics JSON, history CSV, learning-curve image, and test confusion matrix. The LSTM uses packed sequences, so trailing padding is skipped.

## Final experiment

After the smoke test, run the required 20,000 / 5,000 / 25,000 experiment:

```powershell
python .\src\prepare_smoke_data.py --full
python .\src\train_smoke.py --full
```

## Complete team experiment suite

`src/run_all.py` covers the four assigned models and generates a unified final-comparison table:

- Varaang: TF-IDF + one-hidden-layer MLP with dropout
- Rohit: TextCNN
- Arshabrata: BiLSTM
- Navneet: DistilBERT fine-tuning
- all members: IMDB test metrics, Amazon Polarity cross-domain evaluation, character-noise, synonym-substitution, and random-word-deletion robustness tests, plus error-analysis samples

In Colab, choose a GPU runtime, upload this folder to Drive, then run:

```powershell
pip install -r requirements.txt
python src/run_all.py --all
```

Run only one model with `--model mlp`, `--model textcnn`, `--model bilstm`, or `--model distilbert`. Results are written under `results/final/`; every score is measured during that run.

## Integrity note

The smoke-test result is diagnostic, not the report's final metric. Do not alter measured values. The later full run must use the specified 20,000 training, 5,000 validation, and official 25,000-review test split and its actual metrics.

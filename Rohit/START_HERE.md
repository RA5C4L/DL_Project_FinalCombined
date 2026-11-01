# Start here Rohit

You own **TextCNN**, the convolutional sentiment classifier. Your registration number in the supplied documents is 230953168. You are not responsible for implementing MLP, BiLSTM, or DistilBERT.

## Your completed results

The TextCNN is trained and evaluated. Start with `reports/textcnn_seed42/RESULTS.md`, then `reports/textcnn_seed42/ERROR_ANALYSIS.md` and `reports/ROHIT_CONTRIBUTION.md`. Your measured clean IMDB accuracy is 87.08%; Yelp accuracy is 79.96%. The local checkpoint is already saved in `runs/textcnn_seed42/`.

## First session

1. Read `reports/STATUS.md` to see the verified state of this copy.
2. Read `docs/YOUR_PART.md` for the exact assignment and requirements.
3. Follow the README installation commands from this folder, then run the tests and the synthetic smoke command.
4. Inspect `src/rohit_dl/model.py` while reading `docs/VIVA_GUIDE.md`.
5. Use the prepared data and measured runs if present. You do not need to repeat a completed experiment merely to recreate existing evidence.

The synthetic smoke run only checks that the components connect. Its repeated tiny examples and metrics must never appear as IMDB results.

## What to give your team

Share the tokenizer policy, `data/split_indices.json`, `data/manifest.json`, OOD indices, and the actual perturbation files through an agreed local/shared data location. Share the small reports and source code through GitHub later. The shared split is a proposal until your team adopts it; if the MLP owner already has a split, reconcile that before claiming a common-protocol comparison.

Use the measured TextCNN row in the comparison table. Ask teammates for their measured rows, not estimates. Your code cannot establish results for their models.

## How to read the result folders

`best_validation.json` identifies the epoch selected without looking at test scores. `history.json` records training progress. `provenance.json` records data checksums, environment, and parameter count. Each evaluation folder has metrics, predictions, errors, and a confusion matrix. The robustness folder adds `summary.csv` and `robustness.png`.

Before writing the final project conclusions, review examples in `errors.jsonl` and complete the semantic review sheet. Discuss local n-gram features, negation, OOV words, truncation, and domain-specific vocabulary. Do not infer a mechanism solely from a score difference.

## Running on Colab or another machine

Use `notebooks/TextCNN_Walkthrough.ipynb` after placing this project folder in the notebook environment. Select a GPU runtime for full training if available. The notebook uses the package instead of maintaining a second implementation. Save or download the run folder before a temporary runtime ends.

## Later GitHub step

Follow `docs/GITHUB_GUIDE.md`. Dataset files, checkpoints, caches, source Word documents, and local environments are deliberately ignored. The GitHub test workflow downloads packages but does not download the benchmark datasets or run a full training job.

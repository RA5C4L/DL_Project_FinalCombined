# Team handoff

Rohit owns TextCNN. Varaang owns MLP + TF-IDF, Arshabrata owns BiLSTM, and Navneet owns DistilBERT. The final robustness comparison is shared work.

## Shared contract

Each JSONL record contains `id` (unique string), `text` (string), and `label` (0 negative, 1 positive). IMDB IDs use `imdb:train:<original-index>` for train/validation and `imdb:test:<original-index>` for test. Extra perturbation metadata does not change IDs or labels.

Distribute the exact train/validation indices and dataset manifest. Do not independently regenerate a nominally similar split and assume equivalence. Share exported test/corrupted reviews so different tokenizers receive the same underlying text. Do not overwrite an existing team protocol without reconciling it first.

Model-specific feature construction can differ: TF-IDF, sequence vocabulary and transformer tokenization are not interchangeable. Fit all learned preprocessing only on training data. Keep development, final test and OOD roles fixed.

## Result row

Use model, training seed, checkpoint identifier, condition, requested perturbation rate, sample count, accuracy, macro precision, macro recall, macro F1, and accuracy drop in percentage points. State whether the full OOD test or a specified subset was used. Never place projected numbers beside measured numbers without a clear distinction.

TextCNN's `summary.csv` is ready for merging with teammates' measured rows. Prediction JSONL supports per-example paired comparisons. Review false positives and false negatives, including negation, mixed sentiment, uncommon words, truncation and domain-specific vocabulary. Record evidence for each explanation instead of claiming causes from metrics alone.

## Remaining joint decisions

Confirm the shared indices, cross-domain dataset/subset, perturbation rates and implementation, semantic review criteria, and any multi-seed runs. Integrate the three teammate models and final report/presentation in the group repository later. This individual project does not invent their code or results.

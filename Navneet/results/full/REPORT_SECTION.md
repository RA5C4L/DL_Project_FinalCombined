# DistilBERT implementation and measured evaluation

Project owner: Navneet Vivek (230953352), ICT 4442.

The fine-tuned DistilBERT achieved 90.96% accuracy on the complete 25,000-review IMDB test set and 91.71% on the fixed 25,000-review Amazon test sample. The largest observed perturbation-related accuracy drop was 7.24 percentage points for typo_0.2.

## Implementation

The generic pretrained uncased DistilBERT encoder was fine-tuned with a binary classification head. The official IMDB training split was divided into 20,000 training and 5,000 validation examples using stratified sampling with seed 42. The input limit was 256 WordPiece tokens. AdamW used a learning rate of 2e-5, weight decay 0.01, 10% warmup, two epochs and effective batch size 16. The checkpoint with the best validation macro F1 was selected before final test evaluation.

## Measured results

Accuracy, precision, recall and F1 below are percentages. Precision, recall and F1 are macro averages. Drops are percentage points relative to clean IMDB.

| Condition | N | Accuracy | Precision | Recall | Macro F1 | Accuracy drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 25000 | 90.96 | 90.97 | 90.96 | 90.96 | 0.00 |
| amazon | 25000 | 91.71 | 91.82 | 91.71 | 91.70 | -0.74 |
| synonym_0.05 | 25000 | 90.38 | 90.44 | 90.38 | 90.38 | 0.58 |
| synonym_0.1 | 25000 | 89.60 | 89.75 | 89.60 | 89.59 | 1.37 |
| synonym_0.2 | 25000 | 86.49 | 87.17 | 86.49 | 86.43 | 4.48 |
| typo_0.05 | 25000 | 89.88 | 90.02 | 89.88 | 89.88 | 1.08 |
| typo_0.1 | 25000 | 88.24 | 88.65 | 88.24 | 88.21 | 2.72 |
| typo_0.2 | 25000 | 83.72 | 85.34 | 83.72 | 83.53 | 7.24 |
| deletion_0.05 | 25000 | 90.56 | 90.57 | 90.56 | 90.56 | 0.40 |
| deletion_0.1 | 25000 | 90.32 | 90.32 | 90.32 | 90.32 | 0.64 |
| deletion_0.2 | 25000 | 89.10 | 89.11 | 89.10 | 89.10 | 1.86 |

## Error analysis

The clean-test confusion matrix is shown below. Rows are true labels and columns are predictions.

| True label | Predicted negative | Predicted positive |
| --- | ---: | ---: |
| negative | 11447 | 1053 |
| positive | 1206 | 11294 |

For each condition, the accompanying error JSON contains up to 30 confident errors for qualitative inspection. Prediction CSVs preserve review IDs, labels and positive-class probabilities. Perturbed-condition JSONs additionally give prediction flip rate and failure rate among clean-correct examples.

## Interpretation and limitations

These are measured results from one seeded run, not estimates from the interim report. The Amazon result is on a fixed stratified subset of its official test split; no Amazon training data was used. Its comparison with IMDB also reflects differences in review difficulty and length, so it does not isolate a causal effect of domain shift. Random synonym, typo and deletion stress tests were applied at 5%, 10% and 20% rates. Synonym candidates came from nlpaug's WordNet model. These are not optimized TextFooler attacks, and the edits are not guaranteed to preserve meaning or sentiment.

Human semantic review remains pending in `data/perturbed/manual_review.csv`. Until that review is completed, use the term random perturbation robustness rather than certified meaning-preserving adversarial robustness. Truncation can hide later review content and edits. A single model and single seed cannot establish an architecture-level ranking or statistical significance. The group comparison still requires the other members' measured outputs on these same frozen tests.

## Reproducibility

Split SHA-256: `b2aff130e76f84ad088d273a2d4ed30cc7477ae7b9a52fc51f94d428a1667265`.

Inference mode: CUDA float16 autocast. Training hardware: NVIDIA GeForce RTX 3070 Ti Laptop GPU.

See `provenance.json`, `data/manifest.json`, `data/perturbation_manifest.json`, the saved tokenizer/checkpoint, and the environment snapshot for run details.

## Input length analysis

The 256-token limit truncates 40.11% of clean IMDB test reviews and 0.20% of the Amazon sample. This is a measured difference between the inputs, not proof that truncation caused the cross-domain score difference. See `docs/ERROR_ANALYSIS.md` for concrete mixed-sentiment and apparent label-inconsistency examples.

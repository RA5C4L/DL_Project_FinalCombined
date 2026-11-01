# Completed project status

Final packaging and review completed on 31 October 2026.

## Completed

- Implemented Rohit Mohanty's prescribed TextCNN, tokenizer and padding pipeline.
- Prepared the full 20,000 / 5,000 / 25,000 IMDB split and all 38,000 Yelp test reviews.
- Trained for nine epochs on CPU; early stopping selected epoch six by minimum validation loss.
- Evaluated the selected checkpoint on clean IMDB, Yelp, and all nine corruption conditions: 288,000 review instances in total.
- Passed 14 automated tests, including training/checkpoint loading, leakage checks, corruption integrity and report export.
- Checked all four final plots visually and saved per-example predictions and errors.
- Saved actual split indices, dataset revisions, input checksums, resource versions and source checksums.
- Prepared documentation, a guided notebook, Git ignore rules, dependency files and a GitHub test workflow.

## Measured results

| Evaluation | Accuracy | Macro F1 |
|---|---:|---:|
| Clean IMDB, 25,000 reviews | 87.08% | 87.05% |
| Yelp, 38,000 reviews | 79.96% | 79.85% |
| Typos, requested 20% | 82.60% | 82.59% |
| Deletion, requested 20% | 82.65% | 82.58% |

Yelp accuracy drops 7.12 percentage points from clean IMDB. See `textcnn_seed42/RESULTS.md` and the full-precision CSV for all conditions. The interim draft's projected 89.3% accuracy and 89.4% macro F1 are not measured results and should be replaced with the actual values above when updating the TextCNN section.

## Files ready to use

The trained checkpoint and its configuration/vocabulary are in `runs/textcnn_seed42/`. Evaluation predictions, errors and plots are in `runs/robustness_seed42/`. Small shareable reports are in `reports/textcnn_seed42/`. The original course documents are preserved in the ignored `references_local/` folder.

The separate GitHub-ready ZIP contains source, tests, guides, shared protocol metadata and small reports. It intentionally excludes datasets, model weights, caches and the original Word documents. The complete local project folder retains those working assets. No project commit, remote or GitHub push has been made.

## Limits and remaining joint work

This is a single-seed TextCNN experiment. Nine perturbation examples received an AI-assisted qualitative spot check; the full semantic review sheet remains available for independent review. Synonym edits saturate around 4% actual word replacement, even when the requested rate is higher. These are corruption tests, not a validated TextFooler-style adversarial study.

The team must agree on the shared protocol and integrate the other three members' measured results before making architecture comparisons. The code, trained TextCNN, evaluation outputs and Rohit's contribution notes are complete; this package does not invent teammates' results or replace the group's final report and presentation.

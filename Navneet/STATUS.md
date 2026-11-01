# Execution status

Finalized 31 October 2026. The implementation, full training run and all planned automated evaluations are complete. Human semantic review and team integration remain.

- Source implementation and documentation created.
- Python environment installed locally; dependency consistency check passed.
- CUDA verified on NVIDIA GeForce RTX 3070 Ti Laptop GPU.
- 13 automated tests passed, including a tiny-model training/save/reload integration test. The tiny model is a test fixture, not a benchmark result.
- Real IMDB prepared: 20,000 training, 5,000 validation and 25,000 test reviews.
- Real Amazon test-only stratified sample prepared: 25,000 reviews.
- Exact split indices, dataset revisions and fingerprints saved under `data/`.
- Completed two fine-tuning epochs on all 20,000 training reviews. Epoch 1 was selected by validation macro F1 (0.91299); epoch 2's validation macro F1 was lower (0.90937). The selected model and tokenizer are saved in `models/distilbert/`.
- Completed all 11 evaluation conditions, each with 25,000 reviews: clean IMDB, Amazon and nine perturbation conditions.
- Independently checked all 275,000 saved prediction rows against reported confusion matrices, accuracy, macro F1 and accuracy drops. All checks passed; see `results/verification_summary.json`.
- Saved measured result tables, predictions, error examples, provenance and an accuracy chart under `results/full/`. The chart was visually inspected.
- Generated `results/full/REPORT_SECTION.md` from measured outputs. Added concrete error analysis, input-length statistics, viva notes and team handoff instructions.
- Verified the saved model's prediction command on a positive example.
- Human semantic/sentiment review of perturbations remains required before describing them as meaning-preserving.
- No Git initialization, commit or push has been performed.

## Key measured results

| Condition | Accuracy | Drop from clean in percentage points |
| --- | ---: | ---: |
| Clean IMDB | 90.964% | 0.000 |
| Amazon sample | 91.708% | -0.744 |
| Synonyms at 20% | 86.488% | 4.476 |
| Typos at 20% | 83.720% | 7.244 |
| Deletion at 20% | 89.100% | 1.864 |

These are results from one seeded training run. No statistical significance test or multi-seed architecture comparison has been performed. The project demonstrates a useful robustness gap, but does not establish that DistilBERT is better than the other three architectures.

## Remaining work before group submission

1. Review the 450 exported original/perturbed pairs in `data/perturbed/manual_review.csv` for meaning and sentiment preservation. Until then, describe these as random perturbation stress tests.
2. Confirm the team uses these exact split IDs, Amazon sample and perturbation datasets. If their existing split differs, agree on one common split before comparing results.
3. Combine teammates' measured MLP, TextCNN and BiLSTM results with this model's results, and integrate the supplied section into the final group report/presentation.
4. Review the code and viva notes so you can explain the contribution yourself.
5. Commit and push later when the repository destination is agreed, as requested. `.gitignore` excludes environment files, caches, datasets, model weights and original DOCX references.

## Where to start

- `PROJECT_DEVELOPMENT_REPORT.md`: complete project history from the initial documents through implementation, execution and handoff.
- `README.md`: execution and folder guide.
- `docs/YOUR_PART.md`: assignment and requirements mapping.
- `results/full/REPORT_SECTION.md`: measured report contribution.
- `docs/ERROR_ANALYSIS.md`: concrete observations and limitations.
- `docs/GIT_AND_TEAM_HANDOFF.md`: files and identities to share.

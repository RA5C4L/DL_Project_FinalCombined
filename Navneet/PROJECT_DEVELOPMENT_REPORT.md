# Navneet Vivek DistilBERT Project Development Report

**Project:** Robustness of Deep Learning Architectures for Sentiment Classification Under Distribution Shift and Adversarial Text Perturbation  
**Course:** ICT 4442  
**Project member:** Navneet Vivek, 230953352  
**Development period:** 30–31 October 2026  
**Report updated:** 31 October 2026

Navneet's DistilBERT contribution progressed from an unimplemented local starting point to a trained model, a reproducible data pipeline, and completed clean, cross-domain and perturbation evaluations. The selected model achieved 90.964% accuracy on clean IMDB and 91.708% on the fixed Amazon test sample. Its largest measured reduction was 7.244 percentage points under the 20% typo condition. Automated implementation and evaluation are complete; semantic review of perturbations and integration with the other three models remain.

This report records the development history and decisions behind those outputs. The shorter [measured report section](results/full/REPORT_SECTION.md) is available for integration into the group's final academic report.

## Starting point and assigned responsibility

The work began with the supplied project synopsis and Phase 2 interim report, together with Navneet's statement that he was starting his contribution from scratch. No existing DistilBERT implementation, trained checkpoint, shared split file or measured DistilBERT results were supplied for this work. The new `Navneet_DL_Project` folder was created to hold the implementation and its outputs. Git publishing was explicitly deferred.

The synopsis assigns Navneet the fine-tuned DistilBERT model and its evaluation. The interim report additionally assigns DistilBERT environment setup, robustness-library review and the perturbation plan. It describes the DistilBERT setup as scheduled and provides no measured DistilBERT score. Its MLP and TextCNN figures are projected results, so they were not used as measured comparison baselines.

The other model responsibilities remain with the respective team members: MLP with TF-IDF, TextCNN and BiLSTM. This work supplies the transformer component and reusable evaluation assets, rather than implementing those three models or claiming their results.

Development, debugging and execution were carried out with Codex assistance at Navneet's request. The resulting code, measurements and explanations are available for Navneet to inspect, reproduce and understand before submission or viva.

## Requirements taken from the project documents

The implementation retained the interim report's 20,000 training, 5,000 validation and 25,000 test split sizes, stratified sampling with seed 42, minimal cleaning, and common classification metrics. DistilBERT uses the specified 256-token input limit and a two-epoch fine-tuning run, within the report's proposed two-to-three-epoch range.

The 30,000-word learned vocabulary and 400-token sequences described for the other neural models were not applied to DistilBERT. Its pretrained WordPiece tokenizer is necessary to keep input token IDs aligned with its pretrained embeddings.

Several details needed concrete implementation choices: Amazon rather than Yelp for cross-domain testing; a balanced 25,000-review Amazon test sample; perturbation rates of 5%, 10% and 20%; learning rate, batch size and optimization settings; and validation macro F1 for checkpoint selection. These choices are recorded in [config.json](config.json) and the [experimental protocol](docs/EXPERIMENT_PROTOCOL.md). They should be agreed with the group when assembling the four-model comparison.

## Development sequence

### Establishing the workspace and environment

The two source documents were inspected and copied into `references/`. Source modules, tests, documentation, notebooks, data, model and result directories were created under the requested folder.

The default `python` command resolved to a Windows Store alias rather than a working Python installation. A local virtual environment was therefore created using the available bundled Python 3.12 runtime. The computer's NVIDIA GeForce RTX 3070 Ti Laptop GPU was detected, and PyTorch 2.8.0 with CUDA 12.8 support was installed. A CUDA tensor operation verified that the GPU was usable.

Large dependency downloads initially stalled or transferred very slowly. Downloads were retried using ranged requests, and the retrieved wheel files were checked against published SHA-256 hashes before installation. The completed environment passed `pip check`. The portable dependency list is in `requirements.txt`; the tested Windows package versions are in `requirements-lock-windows.txt`.

The local environment is tied to this computer and its Python installation. A copied project should recreate the environment using the instructions in [README.md](README.md), rather than copying `.venv` between machines.

### Preparing the real datasets

The labelled official IMDB training and test partitions were downloaded. The official training partition was divided into 20,000 training reviews and 5,000 validation reviews using stratified sampling with seed 42. The full official 25,000-review test partition was retained. Labels are 0 for negative and 1 for positive.

Cleaning replaces HTML break tags with spaces and collapses repeated whitespace. Negation, punctuation and ordinary word forms are retained. No perturbed reviews enter clean training.

The official Amazon Polarity test partition was used to draw a fixed, stratified 25,000-review sample. Its input text combines the review title and content. Amazon training data was not used for training or model selection. The reported Amazon result represents this sample, not the complete 400,000-review official test partition.

Exact split indices, Amazon sample indices, stable review IDs, dataset revisions and fingerprints were saved. Since no existing team split was provided, these indices were generated locally. The group must either adopt these exact identities or agree on a different shared split and rerun the affected models. Merely choosing seed 42 in different implementations is insufficient to establish identical splits.

### Resolving dataset metadata and fingerprint issues

Restricting the IMDB download to labelled partitions caused the dataset loader to report a missing `unsupervised` partition, because repository metadata described all partitions. The loader was changed to use the explicitly selected files without the full-catalog split check. Explicit checks now enforce the expected 25,000-row IMDB train and test partitions and the 400,000-row Amazon official test partition, followed by additional structural verification of the prepared experiment.

A later verification check found that the dataset library's fingerprints could differ before saving and after loading the saved representation. Provenance recording was corrected to use saved-and-reloaded dataset fingerprints. The existing manifests were updated accordingly; review contents, split membership and labels were not changed by this correction. Subsequent identity and alignment checks passed.

### Implementing and testing training

The model starts from the generic pretrained `distilbert/distilbert-base-uncased` encoder with a newly initialized binary classification head. It does not start from a sentiment-fine-tuned checkpoint. All model parameters are fine-tuned.

An offline integration test first trained a miniature randomly initialized model, saved it, reloaded it and checked prediction. A separate two-step smoke run then exercised the real DistilBERT model and real prepared data. The smoke checkpoint is stored separately and is explicitly rejected by the final evaluator. Neither test is presented as benchmark evidence.

The production run then trained on all 20,000 reviews for two epochs. The settings were:

| Setting | Value |
| --- | --- |
| Maximum input length | 256 WordPiece tokens including special tokens |
| Training batch per device | 8 |
| Gradient accumulation | 2 steps |
| Effective training batch | 16 on one GPU |
| Evaluation batch | 32 |
| Optimizer | AdamW |
| Learning rate | 0.00002 |
| Weight decay | 0.01 |
| Warmup | 10% of training steps |
| Schedule | Linear learning-rate decay |
| Seed | 42 |
| Checkpoint selection | Highest validation macro F1 |
| Training precision | Mixed precision on CUDA |

The recorded Trainer training runtime was 249.17 seconds, approximately 4.15 minutes. This is the training-stage runtime, not total project development time; it excludes environment installation, downloads, separate preprocessing and the final robustness evaluation.

After epoch 1, validation accuracy was 91.30% and macro F1 was approximately 0.91299. After epoch 2, validation accuracy was 90.94% and macro F1 was approximately 0.90937. The pipeline therefore restored the epoch 1 checkpoint, `checkpoint-1250`. The delivered model is that validation-selected checkpoint, even though the planned two training epochs were completed. Test scores were not used to make this selection.

### Building the robustness suite

Three independent perturbation types were generated at each of the three declared rates:

- Synonym replacement uses nlpaug's WordNet candidate model, backed by NLTK, with deterministic custom edit selection.
- Typo generation swaps two unequal adjacent internal characters of an eligible word.
- Word deletion removes eligible words while retaining at least one alphabetic word.

The requested rate is the fraction of alphabetic words targeted, not a fraction of characters. The edit budget is rounded down, capped by eligibility, and recorded alongside the realized rate. Eligible words have at least four characters and exclude an explicit negation protection list. Unchanged examples remain in the evaluation denominator. Per-example seeds derive from the global seed, review ID, perturbation type and rate, so the frozen texts do not depend on which model evaluates them.

All nine conditions contain the same 25,000 original test IDs and labels. The generated test sets can be reused by the other architecture owners. A 450-row audit sheet contains 50 original/altered review pairs per condition for semantic and sentiment review.

These are random perturbation stress tests. They are not optimized TextFooler attacks and are not automatically certified to preserve meaning. WordNet substitutions can be contextually inappropriate, and deletion can remove sentiment-bearing words. Negation protection reduces one risk but does not resolve these limitations.

## Final measured results

The selected model was evaluated on 11 conditions, producing 275,000 prediction records. This total includes repeated evaluation of the same IMDB reviews under different perturbations; it is not a claim of 275,000 independent original reviews. Each condition contains 25,000 records.

Accuracy and macro F1 are percentages below. Accuracy drop is measured in percentage points relative to clean IMDB; a negative drop indicates a higher score.

| Condition | Accuracy | Macro F1 | Accuracy drop |
| --- | ---: | ---: | ---: |
| Clean IMDB | 90.964 | 90.964 | 0.000 |
| Amazon test sample | 91.708 | 91.702 | -0.744 |
| Synonyms at 5% | 90.384 | 90.381 | 0.580 |
| Synonyms at 10% | 89.596 | 89.586 | 1.368 |
| Synonyms at 20% | 86.488 | 86.426 | 4.476 |
| Typos at 5% | 89.884 | 89.876 | 1.080 |
| Typos at 10% | 88.240 | 88.209 | 2.724 |
| Typos at 20% | 83.720 | 83.532 | 7.244 |
| Deletion at 5% | 90.564 | 90.564 | 0.400 |
| Deletion at 10% | 90.320 | 90.320 | 0.644 |
| Deletion at 20% | 89.100 | 89.099 | 1.864 |

The complete precision, recall, F1, confusion matrices, prediction probabilities and per-condition error examples are under `results/full/`. Perturbed-condition JSON files also record prediction flip rates and failure rates among originally correctly classified reviews. Evaluation used CUDA float16 autocast, with that mode recorded in provenance.

## What the results show

For this model and run, increasing perturbation strength reduced accuracy for each of the three tested operations. At the highest tested rate, typos caused the largest reduction, followed by synonym replacement and deletion. This supplies direct evidence for the project's distinction between clean classification performance and performance on altered input.

The Amazon sample scored 0.744 percentage points higher than clean IMDB. This should be reported as observed rather than rewritten as a domain-shift penalty. The two datasets differ in content, difficulty and input length. Token-length analysis found that the 256-token limit truncates 40.108% of clean IMDB test reviews but only 0.200% of the Amazon sample. That is an important observed difference, not proof that truncation caused the score difference.

Inspection of confident errors identified mixed negative descriptions with overall enjoyment, as well as apparent inconsistencies between strongly favourable review text and negative dataset labels. These cases are documented in [ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md). They were not removed or relabelled after inspecting predictions.

This is a substantial course-project contribution: one trained architecture, a complete evaluation pipeline, reproducible test assets and measured robustness findings. It does not establish a new state-of-the-art method, statistical significance, or superiority over MLP, TextCNN and BiLSTM. Those stronger conclusions require additional evidence, including the other models' comparable measurements and, where relevant, repeated training seeds and statistical analysis.

## Verification performed

The automated suite passed 13 tests covering cleaning, metric conventions, split validation, deterministic perturbation behavior, smoke-model rejection, data-manifest rejection and a train/save/reload/predict integration flow. Dependency consistency and source, notebook and PowerShell syntax checks also passed.

The real frozen data passed count, balance, unique-ID, split membership, fingerprint, label-alignment and edit-budget checks. An independent arithmetic check reconstructed confusion matrices, accuracy and macro F1 from all 275,000 saved prediction rows and verified agreement with the JSON and CSV results. Probability ranges, unique IDs and accuracy-drop calculations were also checked.

The accuracy chart was visually inspected. The final model's standalone prediction command was exercised on a positive example and returned a positive prediction. These checks verify execution and output consistency; they do not replace the remaining semantic audit or establish statistical significance.

Evidence is saved in [verification_summary.json](results/verification_summary.json), [data_verification.json](results/data_verification.json), and [reproducibility_checksums.json](results/reproducibility_checksums.json).

## Delivered files and their purpose

| Location | Contents |
| --- | --- |
| `navneet_dl/` | Preparation, training, perturbation, verification, evaluation, prediction, data-statistics and report modules |
| `tests/` | Automated protocol and integration tests |
| `config.json` | Fixed experimental settings |
| `data/` | Shared indices, manifests and locally saved evaluation datasets |
| `models/distilbert/` | Selected trained model, tokenizer and experiment metadata |
| `models/smoke/` | Separate real-model smoke checkpoint, unsuitable for final results |
| `runs/distilbert/` | Training checkpoints, state history and environment snapshot |
| `results/full/` | Measured metrics, predictions, errors, chart, provenance and academic report section |
| `docs/` | Role mapping, experimental protocol, error analysis, robustness review, viva and handoff notes |
| `notebooks/Run_in_Colab.ipynb` | Alternative execution of the same source modules in Colab |
| `run_pipeline.ps1` | Windows pipeline entry point |
| `references/` | Unchanged copies of the supplied project documents |

The original synopsis and interim report were not rewritten. The generated academic report section is intended for later integration into the group's final report. Git was not initialized, and no commit, upload or push was performed. `.gitignore` excludes environments, caches, generated datasets, model weights, training runs and original DOCX references from ordinary Git staging.

## Reproducibility and team integration

The exact model, data and environment details are recorded in `models/distilbert/experiment.json` and `results/full/provenance.json`. The IMDB split SHA-256 is `b2aff130e76f84ad088d273a2d4ed30cc7477ae7b9a52fc51f94d428a1667265`. The pretrained model revision is `12040accade4e8a0f71eabdb258fecc2e7e948be`.

Use [README.md](README.md) for execution instructions and [GIT_AND_TEAM_HANDOFF.md](docs/GIT_AND_TEAM_HANDOFF.md) for the sharing plan. Preserve the completed results before rerunning. The scripts reject several existing output locations to prevent silent replacement of experiment evidence.

For the team comparison, share the exact split indices, Amazon sample indices and frozen perturbed text datasets. The four architectures can use their own encodings while retaining identical review identities, labels and metric definitions. Large datasets and checkpoints can be transferred as artifacts separately from source control.

## Work remaining

1. Complete the 450-pair human semantic and sentiment audit in `data/perturbed/manual_review.csv`. Until then, retain the random stress-test description and its limitations.
2. Confirm that all team members use the same data identities. Resolve any difference from an existing team split before comparing architecture scores.
3. Obtain the other members' measured MLP, TextCNN and BiLSTM outputs and construct the final group comparison.
4. Integrate the measured report section, selected error examples and limitations into the group report and presentation. Navneet should review the implementation and viva notes so he can explain the choices and results.
5. Agree on the repository destination and artifact-sharing arrangement, then commit and push as a separate later step.

Additional seeds, uncertainty estimates, paired significance testing and constrained targeted attacks would strengthen the scientific conclusions. They are extensions beyond the completed single-seed random-perturbation evaluation and have not been represented as completed work.

## Reference records

- [Project synopsis](references/Project%20Synopsis.docx), especially the model-to-member assignment and project objectives.
- [Phase 2 interim report](references/ICT_4442_Phase_2_Interim_Report_Draft.docx), especially preprocessing, model configurations, individual contributions and remaining-work plan.
- [Experimental protocol](docs/EXPERIMENT_PROTOCOL.md), for the implemented definitions and choices.
- [Measured result table](results/full/comparison.csv), for the full-precision evaluation outputs.
- [Execution status](STATUS.md), for the concise completion checklist.

The development history above follows the work carried out in this project session. Metrics and configuration details are supported by the saved run artifacts; no projected numbers from the interim report have been substituted for measured results.

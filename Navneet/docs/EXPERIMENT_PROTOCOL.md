# Experimental protocol

Freeze this protocol before inspecting test results. A change prompted by test performance requires a clearly labelled exploratory experiment; do not present it as the original held-out evaluation.

## Data and preprocessing

Use `stanfordnlp/imdb` official labelled train and test splits. Reserve a stratified 5,000 of the official 25,000 training reviews with scikit-learn `train_test_split(random_state=42)`. Train on 20,000; validate on 5,000; test on all 25,000 official test reviews. Persist the exact indices, dataset fingerprints and split hash. This preserves the official split rather than deduplicating and changing the benchmark.

Replace HTML break tags with spaces and collapse whitespace. Preserve punctuation and negation. No stemming or stopword removal. DistilBERT's uncased tokenizer performs its own normalization. Truncate to 256 WordPiece tokens including special tokens; dynamically pad each batch. Long-review truncation is a limitation, especially if sentiment or perturbations occur beyond the visible prefix.

Use 25,000 stratified Amazon Polarity reviews drawn only from its official test split, with seed 42. Input is title plus content. Save official row IDs. This is a declared subset of the full Amazon test corpus, not a claim to evaluate its entirety. Never use Amazon to train, select epochs or tune settings. Share this subset with the group.

## Model and optimization

Load the generic pretrained `distilbert/distilbert-base-uncased`, not an already sentiment-fine-tuned model. Initialize a two-class classification head. Fine-tune all parameters with AdamW, learning rate 2e-5, weight decay 0.01, linear schedule, 10% warmup, two epochs, per-device batch 8 and gradient accumulation 2. Effective training batch on one GPU is 16. Use automatic mixed precision on CUDA and seed 42. Select highest validation macro F1 from epoch checkpoints; never select using the test set. Exact numerical repeatability across CUDA versions/hardware is not guaranteed, so save environment information.

## Robustness conditions

Run each of three perturbation types independently at 0.05, 0.10 and 0.20 rates. Do not compose edits across conditions. An edit budget is floor(rate × alphabetic word count), capped by eligible words. Words must be alphabetic, at least four characters, and outside the explicit negation protection set. Unchanged examples remain in the denominator. Save realized rates and changed-review counts.

- Synonym: randomly select a lemma alternative from nlpaug's WordNet candidate model; only single alphabetic replacements, with basic case preservation. This is a context-unaware baseline; polysemy and sentiment drift require manual review.
- Typo: transpose two unequal adjacent internal characters, preserving first and last characters.
- Deletion: remove eligible words, retaining at least one alphabetic word.

Per-example random seeds derive from SHA-256 of the global seed, stable review ID, perturbation kind and rate. Generation therefore does not depend on model, batching or evaluation order. Export all frozen texts for teammates. These are random corruption tests motivated by the synopsis, not gradient-based or query-optimized adversarial attacks.

Inspect 50 deterministic samples per condition in `manual_review.csv`. Record meaning and sentiment preservation, reviewer and notes. Do not quietly delete hard examples based on model predictions. Report audit findings and any filtering rule; if a revised suite is needed, freeze and evaluate it for every architecture. Automated negation protection does not establish semantic equivalence.

## Metrics and interpretation

Accuracy = correct / N. Precision, recall and F1 are macro averages over both fixed labels (0 negative, 1 positive), with zero for undefined class ratios. Confusion matrices use true-label rows and predicted-label columns in that order.

Accuracy drop in percentage points = 100 × (clean accuracy − condition accuracy). A negative value means an improvement. Amazon has different examples, so this is an aggregate domain-transfer difference, not a paired error statistic. Perturbed IMDB shares IDs and labels with clean IMDB; also report prediction flip rate and failure fraction among originally correctly classified examples. Failure rate here is not a validated semantic attack success rate.

Batch inference uses float16 autocast on CUDA and float32 on CPU; the actual mode is recorded in evaluation provenance. Keep the inference mode fixed across this model's conditions. Tiny numerical differences can occur between devices or precision modes.

Use the same frozen tests and metrics for all four models. One seeded run does not establish statistical significance or a causal effect of architecture: pretraining, model size, tokenization, sequence length and optimization differ. For stronger conclusions, repeat training seeds and compute paired uncertainty on shared test IDs. The required baseline remains seed 42.

## Completion criteria

Data manifests and split identities saved; production checkpoint trained on all 20,000 reviews; clean, domain and all nine perturbation conditions evaluated; manual audit performed; team comparison uses verified measured numbers; report includes truncation and semantic-preservation limitations. Automated code can create the review sheet but cannot claim a human has completed it.

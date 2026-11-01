# Varaang - text to paste into the Interim Report (auto-generated from mlp_results.json)

## Section 3 - Models table, MLP row
Owner: Varaang Tiwari | Status: Implemented; trained and evaluated on the common test set
Preliminary result: Accuracy 90.3%; macro F1 90.3%
Configuration: 30,000-dim TF-IDF (unigram+bigram) input; one 256-unit ReLU layer; dropout 0.5; Adam (lr 0.001); early stopping (stopped after 5 epochs).

## Section 3 - extra paragraph (optional, under the table)
The MLP baseline was trained on the 20,000-review training split, tuned by early stopping on the 5,000-review validation split, and evaluated once on the 25,000-review official test set. Test accuracy was 90.31% (macro precision 90.32%, macro recall 90.31%, macro F1 90.31%). The confusion matrix was [[11191, 1309], [1113, 11387]] (rows = true negative/positive). The model has 7,680,513 trainable parameters and trained in 117.5 seconds. Results are preliminary and unoptimised; no hyper-parameter search was performed.

## Section 4 - Contribution log row
Varaang Tiwari | 230953378 | IMDB acquisition; shared split file; TF-IDF pipeline; MLP baseline; shared metric script. | Commit history on GitHub (add repo link + commit range) | [signature]

## Section 5 - Risk (MLP-specific, optional)
TF-IDF is vocabulary-bound, so words unseen at training time (typos, rare synonyms) are silently dropped; this is expected to make the MLP sensitive to character noise and cross-domain vocabulary, which Phase 3 will quantify.

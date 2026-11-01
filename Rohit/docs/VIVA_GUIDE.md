# TextCNN explanation and viva preparation

## Explain your contribution

“My assigned model is TextCNN. The pipeline builds the vocabulary from training reviews, converts reviews to padded token sequences, trains a convolutional sentiment classifier, and evaluates it on clean IMDB, cross-domain reviews and controlled text corruptions. I compare macro metrics and accuracy drops, using the same held-out examples as the other models.”

Use that description only after you understand the code and can describe which experiments actually completed. Consult `reports/STATUS.md` for verified evidence.

## Why a convolution works on text

Each filter moves across adjacent word embeddings. Widths 3, 4 and 5 learn local patterns of different lengths. ReLU keeps positive activations, and global max pooling retains the strongest matching pattern anywhere in the visible review. Concatenation combines the three branches. Dropout discourages over-reliance on particular features. The final linear layer assigns logits to negative and positive sentiment.

The model is efficient and naturally suited to local phrases. It can lose information about the order of distant phrases after pooling. Long-distance negation, mixed reviews and context outside the first 400 tokens may be difficult. These are hypotheses to examine using errors, not automatic conclusions.

## Questions to prepare

- **Why use validation data?** To select the checkpoint and any development choices without tuning against final test labels.
- **Why a training-only vocabulary?** Validation and test word frequencies are information about held-out data; fitting on them would leak information.
- **Why macro F1?** It gives equal weight to each class's F1. Accuracy alone can conceal class-specific mistakes.
- **Why no softmax before cross-entropy?** PyTorch cross-entropy expects logits and performs the appropriate normalization internally.
- **Why protect “not”?** Removing a negator can reverse the target sentiment. Protection reduces one problem but does not guarantee semantic equivalence.
- **Why does typo noise matter?** A changed word can become OOV and break a learned local pattern.
- **Why OOD data only at evaluation?** The question is transfer under distribution shift without adapting to the target labels.
- **Is this TextFooler?** No. These are seeded random corruption tests, not a query-based semantic attack search.
- **What is accuracy drop?** Clean accuracy minus shifted/corrupted accuracy, multiplied by 100 for percentage points.
- **Is 89.3% your result?** It was a projection in the interim draft. Cite only the actual measured run outputs.

## Before presenting

Open a correctly predicted review and a confident error. Trace their preprocessing, token IDs, padding and prediction. Explain the input/output shapes in `model.py`. Show the saved split indices, best-validation epoch, provenance, confusion matrix and robustness curve. Discuss single-seed and semantic-review limitations honestly.

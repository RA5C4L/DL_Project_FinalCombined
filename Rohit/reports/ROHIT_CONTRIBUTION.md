# Rohit Mohanty TextCNN contribution

Rohit Mohanty, registration number 230953168, is assigned TextCNN implementation and evaluation in the project synopsis. The interim report additionally names tokenizer/padding and preliminary evaluation. Robustness evaluation and the final comparison are shared team tasks.

## Implementation and measured work

This contribution provides a reproducible PyTorch TextCNN pipeline. It uses a training-only vocabulary capped at 30,000 entries, 400-token inputs, trainable 128-dimensional embeddings, 128 convolution filters at each of widths 3, 4 and 5, global max pooling and dropout 0.5. The official IMDB training portion is split into 20,000 training and 5,000 validation examples with stratification and seed 42; the official 25,000 test examples remain held out.

The full model trained for nine epochs, with epoch six selected by minimum validation loss. It achieved 87.08% accuracy and 87.05% macro F1 on IMDB. On all 38,000 Yelp test reviews, accuracy was 79.96% and macro F1 was 79.85%, a 7.12-percentage-point accuracy drop. These observed figures replace the interim draft's projections for the TextCNN contribution.

The contribution also includes three corruption families at requested rates of 5%, 10% and 20%, with fixed seeds, persisted reviews, actual edit-rate reporting and qualitative inspection. At the highest requested rate, typo and deletion accuracy was 82.60% and 82.65%. Conservative synonym substitutions reached only about 4% realized edits, so their smaller drop cannot be interpreted as an equal-intensity robustness advantage.

## Evidence and integration

The package includes a trained checkpoint, configuration, vocabulary, training history, prediction/error files, confusion matrices, robustness and learning curves, a data audit, 14 passing tests, a guided notebook, viva notes, and GitHub preparation files. Small reports and shared split metadata are ready for the group repository; large data and checkpoint files remain local and ignored by Git.

The experiment covers one model and one training seed. The team still needs to reconcile the common protocol, independently review semantic validity as needed, and merge measured results from MLP, BiLSTM and DistilBERT. No result for another member's model is asserted here. Follow course requirements for acknowledging assistance, and use this note only after understanding the code and evidence.

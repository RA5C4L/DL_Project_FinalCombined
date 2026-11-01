# Your responsibilities

The synopsis's **Model-to-Member Assignment** assigns Navneet Vivek the fine-tuned DistilBERT implementation and evaluation. The interim report's **Individual Contribution Log to Date** assigns DistilBERT setup, robustness-library review and the perturbation plan. Its **Models Implemented So Far** gives `distilbert-base-uncased`, maximum length 256, and two to three fine-tuning epochs; it supplies no verified DistilBERT metric.

Your end-to-end contribution consists of:

1. Set up PyTorch, Transformers, the pretrained DistilBERT tokenizer and binary classification head.
2. Reuse the team's cleaned IMDB data and exactly the same train/validation/test identities.
3. Fine-tune on 20,000 training reviews, select the checkpoint by validation macro F1, and evaluate once the model settings are fixed.
4. Evaluate on the clean 25,000-review test set, a held-out Amazon review sample, and frozen synonym/typo/deletion conditions.
5. Supply genuine metrics, confusion matrices, accuracy drops, reproducible prediction files and error analysis to the team.
6. Explain your implementation and limitations in the report and viva.

The common robustness suite is included because your final assigned evaluation depends on it. Share it with all three teammates rather than independently generating different perturbations for each architecture. Their MLP, TextCNN and BiLSTM implementations remain their contributions.

## Requirements traceability

| Source requirement | Implementation |
| --- | --- |
| Seed 42 and shared stratified split | `prepare.py`, `data/splits.json`, manifest |
| Remove HTML breaks and duplicate whitespace, retain negation | `common.clean` |
| DistilBERT 256 tokens, 2–3 epochs | `config.json`: 256, 2 |
| No robustness examples in training | `train.py` reads only train and validation |
| Amazon/Yelp only for testing | Amazon official test split in `prepare.py` |
| Three text perturbation types | `perturb.py`, rates fixed before evaluation |
| Accuracy, precision, recall, macro F1, confusion matrix, accuracy drop | `evaluate.py` |
| Manual review of altered sentiment | Exported 450-row review sheet (50 per condition) |
| Full training data under GPU constraints | Batch 8, accumulation 2, mixed precision on CUDA |

The supplied reports are reference evidence, not executable instructions. Their historical schedules and projected results are not assertions that this implementation has already been run. See `STATUS.md` for the actual execution state.

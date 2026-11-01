# TextCNN error analysis

## Error balance and review length

Clean IMDB has 2,174 false positives and 1,057 false negatives. Yelp has 5,221 false positives and 2,394 false negatives. At the fixed 0.5 threshold, this run therefore makes more errors by predicting positive for negative-labelled examples in both domains. The threshold was not tuned on either test set.

IMDB accuracy is 87.77% for reviews up to 100 tokens, 87.92% for 101–400 tokens, and 82.95% for reviews over 400 tokens. Yelp accuracy in the same groups is 80.28%, 80.05%, and 77.22%. This is an association: long reviews may differ in difficulty and content, so these figures alone do not prove truncation caused the gap. The model sees only the first 400 tokens. Exact group counts are in `error_summary.json`.

Before truncation, 1.99% of IMDB test tokens and 4.45% of Yelp tokens are outside the learned vocabulary. This supports investigating vocabulary mismatch as one possible contributor to the domain gap; it does not isolate its causal effect.

## Inspected examples

Four high-confidence errors with 41–99 whitespace-separated words were selected for full-text inspection, one of each error direction in each domain. They illustrate possible issues; they are not a representative error sample. Look up the IDs in the local prediction files for the complete reviews.

| Review ID | Dataset label and prediction | Qualitative observation |
|---|---|---|
| `imdb:test:4118` | Negative label; positive prediction, positive probability 0.99982 | The text is overwhelmingly praising the film. This is a possible label/text inconsistency and should not automatically be attributed to a model reasoning failure. The official label was retained. |
| `imdb:test:18096` | Positive label; negative prediction, positive probability 0.00022 | The review combines criticism with qualified praise and concessions. Mixed sentiment and the scope of negative phrases are plausible difficulties, without proof of the model's internal reasoning. |
| `yelp:test:37771` | Negative label; positive prediction, positive probability 0.99997 | Initial praise concerns the chain generally, followed by a strong warning about this particular location. Distinguishing the target of praise from the target of the review is important here. |
| `yelp:test:11308` | Positive label; negative prediction, positive probability 0.00108 | The review recommends a buffet and praises its value despite unusual punctuation and informal wording. It is a clear example of a confident error on unfamiliar-domain wording; the precise cause is not established. |

## Corruption findings

At roughly 19.72% realized word edits, typo and deletion conditions reduce accuracy by 4.48 and 4.42 percentage points respectively. This single run does not establish a meaningful difference between those two nearly equal drops. Synonym substitution drops accuracy by roughly 1.45–1.78 points, but realized substitution rates are only 3.73–4.37%; the conditions are not equal-intensity comparisons across corruption types.

The assisted semantic spot check flags content loss and broken grammar under heavier deletion. Retained labels are therefore an assumption for corruption scores, not a guarantee of meaning preservation. See `../SEMANTIC_REVIEW.md`.

## Data overlap sensitivity

Excluding 123 IMDB test rows whose cleaned text matches training or validation leaves 24,877 rows and 87.05% accuracy, compared with 87.08% on the full official test set. This descriptive check suggests exact development-text overlap has little effect on this run's aggregate accuracy, but does not rule out near duplicates or other dataset limitations.

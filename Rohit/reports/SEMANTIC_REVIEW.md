# Qualitative corruption inspection

Nine final-version examples were inspected with AI assistance: three corruption rates each for three different short reviews. The reviews were selected from the shortest items in the seeded audit sample to permit full-text inspection. This is a convenience spot check, not a representative validation study or human inter-annotator agreement exercise.

The synonym examples retain positive sentiment and readable wording. The typo examples remain clearly negative and readable despite the intended spelling errors. Deletion at 10 percent leaves positive sentiment but breaks grammar; the 20-percent deletion example loses enough content that meaning preservation is uncertain. These observations support describing the results as corruption robustness, without asserting that every perturbation preserves the benchmark label.

The full audit sheet is in `data/perturbed/semantic_review.csv`: nine entries have assisted notes and 261 remain available for independent review. The compact `semantic_spot_check.csv` contains the inspected IDs and judgments without republishing review text. No examples were removed from the benchmark based on these judgments.

## Actual edit intensity

Mean synonym edit rates are approximately 3.73%, 4.34%, and 4.37% for requested rates 5%, 10%, and 20%. The conservative adjective/adverb candidate pool saturates. A flat synonym curve at higher requested rates must not be interpreted as robustness to 20% actual word replacement. Typo and deletion rates are approximately 4.68%, 9.70%, and 19.72%. Use the exact realized rates and unchanged counts in the saved manifest.

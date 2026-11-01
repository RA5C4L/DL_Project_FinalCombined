# Inspecting measured errors

These observations come from the saved clean and Amazon prediction/error files for this run. The examples are selected from the most confident mistakes, so they illustrate failure modes rather than estimate how frequent those modes are.

## Mixed praise and negative descriptions

`imdb:test:14565` has a positive dataset label but was classified negative, with positive probability about 0.0081. The reviewer gives the film a high rating because its poor production makes it entertaining. Most of the review describes bad acting, implausible action and poor filming. This is a case where literal negative descriptions and the reviewer's final enjoyment point in different directions. It illustrates the difficulty of interpreting ironic or deliberately contradictory reviews. Truncation is an additional possible limitation for this long review, not an established cause of this prediction.

## Apparent text and label inconsistencies

`imdb:test:12061` has a negative dataset label, but its text is strongly favourable and explicitly gives a high rating. The model predicts positive with probability about 0.9917. This warrants source-label inspection. It should not automatically be interpreted as the model misunderstanding the expressed sentiment, and the benchmark label should not be changed after seeing the prediction.

Two similarly confident Amazon examples, `amazon:test:109008` and `amazon:test:76686`, have negative dataset labels but describe books favourably. Both were predicted positive with probability around 0.99. These examples suggest that apparent label/text inconsistencies can contribute to measured errors in both corpora; inspecting a few examples does not establish a dataset-wide annotation error rate.

## Using the evidence

The saved tokenizer-length analysis found a median of 219 tokens for clean IMDB and 90 for Amazon (including special tokens). The 256-token cutoff truncates 40.108% of IMDB test reviews and 0.200% of the sampled Amazon reviews. This is a measured input-distribution difference that may help explain why the Amazon score is not lower; it does not establish truncation as the cause of any individual error or of the aggregate difference. The model settings remain unchanged after observing these statistics.

Use the confusion matrices to assess class asymmetry and the full prediction CSVs to measure rates. Use the error examples for qualitative discussion. Do not silently remove examples, relabel the test set, tune the checkpoint, or assert that every high-confidence mistake is annotation noise.

For the perturbation conditions, compare the same review ID with the clean prediction and inspect the corresponding original/altered text. The generated manual review sheet must still be completed before claiming meaning-preserving adversarial evaluation. A confidence score is a softmax output, not a calibrated probability of correctness.

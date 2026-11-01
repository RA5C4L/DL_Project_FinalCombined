# Rohit Mohanty project responsibilities

The synopsis assigns Rohit Mohanty the TextCNN implementation and evaluation. The interim report's Section 4 expands that to tokenizer and padding, TextCNN implementation, and preliminary evaluation. Section 5 assigns cross-domain and three perturbation tests and final comparison/error analysis to all members.

| Requirement and source | Implementation |
|---|---|
| Synopsis model-to-member table: TextCNN | `src/rohit_dl/model.py` |
| Interim Section 2: clean breaks/whitespace; preserve negation | `text.clean_text` and `text.tokenize` |
| Interim Section 2: seed 42, stratification, 20k/5k/25k | `data.prepare`, persisted split indices and manifest |
| Interim Section 2: training-only 30k vocabulary, OOV, length 400 | `text.build_vocab` and `text.encode` |
| Interim Section 3: embedding 128, 128 filters at 3/4/5, pooling, dropout .5 | `TextCNN` and `configs/baseline.json` |
| Synopsis: train and test on IMDB | `engine.train` and `engine.evaluate` |
| Synopsis: Amazon or Yelp only for OOD testing | `prepare --ood yelp` (default), optional Amazon |
| Synopsis: synonyms, character noise, word deletion | `perturb.generate`, saved review sets and audit sheet |
| Interim Section 2: common metrics and drops | `engine.metrics`, `robustness.run_suite` |
| Interim Section 5: final comparison and error analysis | Report exports and team handoff guide |

## Choices not specified by the documents

PyTorch, random trainable embeddings, lowercase regex tokenization, PAD/UNK included in the vocabulary cap, Adam at 0.001, batch size 64, at most 15 epochs, patience 3, gradient clipping at 5, minimum validation loss, and requested perturbation rates 5/10/20 percent are explicit implementation choices. They are not quoted requirements. Team agreement is needed before claiming they are the protocol used by all four architectures.

The synonym generator uses the nlpaug WordNet candidate backend suggested by the synopsis, restricted to contextually tagged base adjectives/adverbs and the first sense for that part of speech, with deterministic edit selection and a review export. Character swaps and deletions use explicit seeded functions so edit counts and protected words remain auditable. This is a corruption benchmark, not a reproduction of TextFooler. Short tokens, function words, negation, contractions and names are protected to reduce obvious meaning and grammar changes; human review is still necessary.

## What the documents do not prove

The interim report calls the model implementation-ready with run and verification pending. The 89.3% accuracy and 89.4% macro F1 values are projected. This repository starts from scratch and does not assert that Rohit previously ran or personally wrote any earlier implementation. Review the generated code and follow any course rules for acknowledging assistance.

The final submission date in both supplied plans is 31 October 2026. Do not backdate new experiments to match earlier milestones.

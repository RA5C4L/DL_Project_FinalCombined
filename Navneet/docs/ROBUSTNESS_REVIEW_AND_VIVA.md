# Robustness library review and viva notes

The synopsis suggests nlpaug and/or TextAttack. This implementation uses nlpaug's WordNet candidate model (backed by NLTK) for synonyms, with a small inspectable perturbation layer to make budgets, row IDs and random seeds explicit. Typo and deletion operations are implemented directly. It does not use TextAttack or claim to reproduce TextFooler.

`nlpaug` provides augmentation operations suited to random corruptions. `TextAttack` provides complete attack recipes and model wrappers, including search strategies and constraints for targeted misclassification. A random synonym replacement should not be called TextFooler. Adding TextFooler would require the same attack recipe, search budget and semantic constraints for every architecture, and separate reporting of query budgets and success on originally correct examples. Avoid mixing such results with random corruption rates.

The current WordNet substitution is deliberately labelled a stress-test baseline. It lacks part-of-speech matching and contextual semantic similarity checks. It can change meaning, and all three perturbations can affect sentiment. The exported review sheet is required evidence for any meaning-preservation claim.

## Questions you should be able to answer

**What is your model?** A pretrained six-layer DistilBERT encoder with a two-class classification head, fine-tuned end-to-end on IMDB. Knowledge distillation makes it smaller than BERT while retaining much of its language representation capability.

**Why not build the 30,000-word vocabulary?** That setting belongs to the CNN/BiLSTM. DistilBERT must use the pretrained WordPiece tokenizer to keep token IDs aligned with pretrained embeddings.

**Why 256 tokens?** The interim report specifies it for memory efficiency. The consequence is losing later content in long reviews, which should be acknowledged.

**What prevents leakage?** Train and validation come only from the official training split; checkpoint selection uses validation macro F1. Test and Amazon data never enter the training loop or model selection. Perturbations are generated only from official IMDB test reviews.

**Why macro F1 as well as accuracy?** It gives both classes equal weight and exposes asymmetric precision/recall. The full confusion matrix explains the errors behind aggregate scores.

**Does a stronger clean score prove greater robustness?** No. Compare accuracy drops and actual shifted/perturbed scores. A model can start higher but lose more accuracy.

**Are these attacks guaranteed to preserve meaning?** No. These are controlled random stress tests. Negation protection reduces one source of damage; manual auditing is still needed.

**What belongs in your conclusion?** Only observations supported by measured results. Discuss concrete error cases, changes in performance and limitations; do not claim an architecture ranking before the other models have been run comparably.

## References

- Sanh et al., DistilBERT: https://arxiv.org/abs/1910.01108
- Jin et al., TextFooler: https://arxiv.org/abs/1907.11932
- TextAttack documentation: https://textattack.readthedocs.io/
- nlpaug project: https://github.com/makcedward/nlpaug
- NLTK WordNet: https://www.nltk.org/howto/wordnet.html

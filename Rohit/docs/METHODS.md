# TextCNN methods and experiment protocol

## Data separation

Load the Hugging Face `stanfordnlp/imdb` official train and test splits. Stratify the 25,000 official training examples with scikit-learn, validation size 5,000, seed 42. Persist original row indices; sort them for stable output order. Train on the remaining 20,000 examples. Leave all 25,000 official test examples for final evaluation. Record dataset fingerprints and SHA-256 hashes of each exported split.

Fit vocabulary counts on the training texts only. HTML break tags become spaces, HTML entities are decoded, and repeated whitespace is collapsed. Tokenization lowercases, preserves contractions, and treats punctuation as tokens. Use the most frequent 29,998 tokens with lexical tie-breaking plus PAD=0 and UNK=1. Cap input at the first 400 tokens and right-pad shorter reviews.

Yelp test is the default OOD evaluation. Amazon is supported. No target-domain examples participate in training or model selection. A requested subset is stratified, seeded and saved; label mapping is negative=0, positive=1. All models must use the same source splits and OOD selection for comparison.

## Architecture and optimization

An embedding lookup produces a batch × 400 × 128 tensor. Parallel convolutions use widths 3, 4, and 5, each with 128 filters. Apply ReLU and global max pooling to each branch; concatenate to 384 features; apply dropout 0.5; output two class logits. Cross-entropy receives logits directly. Softmax is used only to produce probabilities for inference.

Mask padding-only and incomplete trailing windows during pooling. For reviews shorter than a kernel, use the first padded window; this also makes empty inputs finite. Train the embedding with the model and keep the padding embedding fixed at zero.

Use Adam with learning rate 0.001, no weight decay, batch size 64, gradient norm cap 5, and up to 15 epochs. Select minimum validation cross-entropy, stopping after three non-improving epochs. Test and OOD data are not loaded by the training function. Seeds are fixed for Python, NumPy and PyTorch. Exact results can vary across devices and software versions; deterministic algorithms are requested with warnings rather than hard failure.

## Robustness conditions

For each test review and each requested rate 0.05, 0.10, 0.20, derive a deterministic random generator from seed, review ID, condition and rate. Request floor(rate × number of word tokens) edits, capped by available eligible words. This definition includes protected words in the denominator and can produce a lower realized rate. Export both edit counts and realized rates.

Synonyms come from the nlpaug WordNet backend, restricted to the first WordNet sense for the contextual part of speech. Only base-form adjectives (JJ) and adverbs (RB) are eligible, and replacements must be one alphabetic word. NLTK POS tagging reduces noun/verb sense mistakes; the restricted candidate pool often makes realized rates lower than requested rates. Typos swap unequal adjacent internal characters. Deletion removes selected words while keeping at least one word. Explicit negators, contractions, tokens shorter than four letters, scikit-learn English stop words, uppercase acronyms and non-initial capitalized/mixed-case names are protected. Apply edits to original word spans after shared cleaning so punctuation spacing is not changed merely by token reconstruction. Generate the full review first; the model's normal 400-token cap is applied during encoding. Thus some edits may lie beyond the model's visible input. Report this limitation when interpreting rates.

These are random model-independent corruptions, not optimized adversarial attacks. A synonym can be inappropriate in context, and removing an adjective can remove sentiment. The exported sample sheet must be checked for sentiment preservation and grammar. For a later controlled attack study, add contextual semantic constraints and a separately specified attack/search budget.

## Metrics and comparisons

Accuracy is the number of correct predictions divided by N. Precision, recall, and F1 are macro-averaged over the two labels with zero division mapped to 0. Confusion matrices have actual labels in rows and predictions in columns, ordered negative then positive. Probabilities >=0.5 predict positive.

Accuracy drop in percentage points = 100 × (clean accuracy − condition accuracy). A negative drop is an improvement. Cross-domain samples are different examples, so this is a domain-level difference, not a paired statistic. For corruptions, additionally report the fraction of originally correctly classified reviews that become incorrect; this is not a certified adversarial success rate because semantic validity is unverified.

The suite checks clean-input checksum, perturbed-file checksums, ordered IDs and labels, and uses one checkpoint throughout. Preserve prediction-level outputs for paired analysis. Multiple training seeds and uncertainty estimates are recommended before making strong architecture-level claims.

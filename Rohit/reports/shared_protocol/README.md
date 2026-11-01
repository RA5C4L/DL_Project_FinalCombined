# Shared protocol files

These are the actual indices and file checksums used by the TextCNN run. Commit these small files with the code so teammates can adopt or compare the split explicitly. Raw data and checkpoints remain ignored.

The split indices refer to the original Hugging Face dataset row order recorded by the dataset fingerprints. The manifest gives SHA-256 checksums of the exported JSONL files, not hashes of the original downloads. The OOD indices select all 38,000 Yelp test examples in this run.

The perturbation manifest identifies each exact corruption file. It does not recreate the reviews by itself; use the fixed generator version and seed or share the generated files through an agreed data location.

`dataset_revisions.json` records resolved Hugging Face commits. `perturbation_resources.json` records the downloaded WordNet and POS-tagger archive checksums used by the final corruption generator.

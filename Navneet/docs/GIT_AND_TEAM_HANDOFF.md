# Git and team handoff

Nothing has been initialized, committed or pushed. When the team is ready, copy this folder into the agreed repository location or initialize it there. Confirm the repository's existing layout before merging.

Commit source, tests, config, notebooks, documentation, `requirements.txt`, `.gitignore`, split/index files and manifests. Commit compact measured summary metrics and the final chart after inspection. Predictions are useful reproducibility artifacts but can be uploaded separately if the repository is too large.

Do not commit `.venv`, downloaded datasets, Hugging Face cache, training runs, large model weights or the supplied documents. `.gitignore` excludes those. Use an agreed artifact store or Git LFS later for weights if required. Review error excerpts before publishing raw review text. There are no credentials in the project.

Give teammates `data/splits.json`, `data/amazon_test_indices.json`, `data/manifest.json` and the frozen `data/perturbed/` dataset folder via local/cloud artifact transfer. They must align predictions by the stable `id`, not assume arbitrary row ordering. All models should use the same text cleaning and frozen datasets; their tokenizer encodings differ by design.

Their comparison rows should use `model,condition,n,accuracy,precision_macro,recall_macro,f1_macro,accuracy_drop_pp` with metrics stored as fractions, drops as percentage points. Use the same label order. An independently regenerated seed-42 split is insufficient proof of identity: compare the saved split hash.

Before pushing: run tests, confirm `STATUS.md` is accurate, inspect staged filenames and size, remove accidental caches or credentials, and replace projected report numbers only with actual measured results. This preparation does not authorize publishing; the user explicitly deferred Git.

# Model storage
Training saves `best.pt`, vocabulary, configuration, and provenance under `runs/<run-name>/`.
Both models and runs are ignored by Git. Keep all those files together for inference.
Use a release attachment or agreed model store later if sharing trained weights; avoid committing large binaries.

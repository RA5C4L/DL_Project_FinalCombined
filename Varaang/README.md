# ICT 4442 - Varaang's part: MLP + TF-IDF baseline

Run from the project root:

    pip install -r requirements.txt
    python scripts/01_prepare_data.py     # creates splits/imdb_split.json (shared by ALL members)
    python scripts/02_train_mlp.py        # trains, tests once, writes results/ + report_snippet.md

Outputs in results/: mlp_results.json, mlp_training_curves.png, mlp_confusion_matrix.png,
report_snippet.md, tfidf_vectorizer.joblib, mlp_best.pt, mlp_test_predictions.npy

LLM disclosure: the initial scaffold of these scripts was drafted with Claude (Anthropic)
and then run, checked and understood by the author.

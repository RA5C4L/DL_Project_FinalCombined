# Local datasets
Generated data is ignored by Git. Run `rohit-dl prepare --output data --ood yelp` from the project root.
Share `split_indices.json`, `ood_indices.json`, and checksums with teammates so all four models use identical examples.
Labels: 0 negative, 1 positive. Never fit a tokenizer on validation, IMDB test, or OOD data.
The raw reviews remain local. Review the dataset's terms before redistribution.

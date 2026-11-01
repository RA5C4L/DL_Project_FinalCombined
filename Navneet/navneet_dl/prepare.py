"""Acquire only official labelled IMDB and Amazon test splits; persist row identities."""
import argparse
from datasets import DatasetDict, load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from .common import ROOT, clean, config, digest, read_json, write_json


def validate_splits(splits, labels):
    train, val = splits["train"], splits["validation"]
    if len(train) != 20000 or len(val) != 5000:
        raise ValueError("Expected 20,000 train and 5,000 validation indices")
    if len(set(train + val)) != 25000 or set(train + val) != set(range(25000)):
        raise ValueError("Indices must be disjoint and cover the official training split")
    for name, ids in [("train", train), ("validation", val)]:
        if sum(labels[i] for i in ids) != len(ids) // 2:
            raise ValueError(f"{name} must be balanced")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", help="Team JSON with train/validation official train row indices")
    args = parser.parse_args()
    cfg = config()
    destination = ROOT / "data/prepared"
    if destination.exists():
        raise FileExistsError("Prepared data exists; use it rather than silently replacing the experiment")
    from huggingface_hub import HfApi
    api = HfApi()
    imdb_revision = api.dataset_info("stanfordnlp/imdb").sha
    amazon_revision = api.dataset_info("fancyzhx/amazon_polarity").sha
    imdb = load_dataset("stanfordnlp/imdb", revision=imdb_revision,
                        data_files={"train": "plain_text/train-*.parquet",
                                    "test": "plain_text/test-*.parquet"}, verification_mode="no_checks")
    train, test = imdb["train"], imdb["test"]
    # Repository metadata expects the unused unsupervised split too. Validate
    # our explicitly selected official partitions rather than that full catalog.
    if len(train) != 25000 or len(test) != 25000:
        raise ValueError("Unexpected official IMDB split size")
    labels = list(train["label"])
    if args.split_file:
        splits = read_json(args.split_file)
    else:
        tr, va = train_test_split(list(range(len(train))), test_size=5000,
                                  random_state=cfg["seed"], stratify=labels)
        splits = {"train": tr, "validation": va}
    validate_splits(splits, labels)
    train = train.map(lambda row, i: {"text": clean(row["text"]), "id": f"imdb:train:{i}"}, with_indices=True)
    test = test.map(lambda row, i: {"text": clean(row["text"]), "id": f"imdb:test:{i}"}, with_indices=True)
    amazon = load_dataset("fancyzhx/amazon_polarity", revision=amazon_revision,
                          data_files={"test": "amazon_polarity/test-*.parquet"}, split="test",
                          verification_mode="no_checks")
    if len(amazon) != 400000:
        raise ValueError("Unexpected official Amazon test split size")
    amazon_ids, _ = train_test_split(list(range(len(amazon))), train_size=cfg["amazon_test_size"],
                                      random_state=cfg["seed"], stratify=amazon["label"])
    amazon = amazon.select(amazon_ids).map(lambda row, i: {
        "id": f"amazon:test:{amazon_ids[i]}",
        "text": clean(row["title"] + " " + row["content"])
    }, with_indices=True).select_columns(["text", "label", "id"])
    data = DatasetDict(train=train.select(splits["train"]),
                       validation=train.select(splits["validation"]), test=test, amazon=amazon)
    data.save_to_disk(str(destination))
    # Datasets may adjust its fingerprint when restoring formatting on load.
    # Record the persisted representation used by training and verification.
    data = load_from_disk(str(destination))
    write_json(ROOT / "data/splits.json", splits)
    write_json(ROOT / "data/amazon_test_indices.json", amazon_ids)
    write_json(ROOT / "data/manifest.json", {
        "seed": cfg["seed"], "split_hash": digest(splits),
        "dataset_revisions": {"stanfordnlp/imdb": imdb_revision, "fancyzhx/amazon_polarity": amazon_revision},
        "dataset_fingerprints": {k: v._fingerprint for k, v in data.items()},
        "counts": {k: len(v) for k, v in data.items()},
        "labels": {"0": "negative", "1": "positive"},
        "amazon_input": "title + space + content", "amazon_sample_hash": digest(amazon_ids),
        "split_origin": args.split_file or "Generated locally; share with team before comparing models"})


if __name__ == "__main__":
    main()

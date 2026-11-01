"""Prepare common, persisted splits; OOD data is never used for fitting."""
from pathlib import Path
from sklearn.model_selection import train_test_split
from .io import save_json, write_jsonl, sha256


def split_indices(labels, seed=42, validation_size=5000):
    train, val = train_test_split(list(range(len(labels))), test_size=validation_size,
                                    random_state=seed, stratify=labels)
    return sorted(train), sorted(val)


def prepare(output="data", seed=42, ood="yelp", ood_limit=0, revision=None):
    from datasets import load_dataset
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f).exists() for f in ("train.jsonl", "validation.jsonl", "test.jsonl")):
        raise FileExistsError("Prepared data exists; choose a new directory to preserve shared splits")
    kw = {"revision": revision} if revision else {}
    train = load_dataset("stanfordnlp/imdb", split="train", **kw)
    test = load_dataset("stanfordnlp/imdb", split="test", **kw)
    if len(train) != 25000 or len(test) != 25000:
        raise ValueError("Expected official IMDB 25,000/25,000 splits")
    tr, va = split_indices(train["label"], seed)
    def records(ds, indices, prefix):
        for i in indices:
            yield {"id": f"{prefix}:{i}", "text": ds[i]["text"], "label": int(ds[i]["label"])}
    write_jsonl(output/"train.jsonl", records(train, tr, "imdb:train"))
    write_jsonl(output/"validation.jsonl", records(train, va, "imdb:train"))
    write_jsonl(output/"test.jsonl", records(test, range(len(test)), "imdb:test"))
    save_json(output/"split_indices.json", {"seed": seed, "dataset": "stanfordnlp/imdb",
        "train": tr, "validation": va, "test": list(range(len(test))),
        "source_train_fingerprint": train._fingerprint, "source_test_fingerprint": test._fingerprint,
        "requested_revision": revision})
    metadata = {"seed": seed, "label_mapping": {"0": "negative", "1": "positive"},
        "train_count": len(tr), "validation_count": len(va), "test_count": len(test),
        "ood": None}
    if ood != "none":
        name = {"yelp": "fancyzhx/yelp_polarity", "amazon": "fancyzhx/amazon_polarity"}[ood]
        ds = load_dataset(name, split="test")
        indices = list(range(len(ds)))
        if ood_limit:
            if ood_limit < 2 or ood_limit >= len(ds):
                raise ValueError("OOD limit must be 0 (all) or between 2 and test-size minus 1")
            _, indices = train_test_split(indices, test_size=ood_limit, random_state=seed, stratify=ds["label"])
            indices = sorted(indices)
        rows = []
        for i in indices:
            item = ds[i]
            text = item.get("text") or " ".join([item.get("title", ""), item.get("content", "")]).strip()
            rows.append({"id": f"{ood}:test:{i}", "text": text, "label": int(item["label"])})
        write_jsonl(output/"ood.jsonl", rows)
        save_json(output/"ood_indices.json", indices)
        metadata["ood"] = {"dataset": name, "split": "test", "count": len(rows), "fingerprint": ds._fingerprint}
    metadata["sha256"] = {p.name: sha256(p) for p in sorted(output.glob("*.jsonl"))}
    save_json(output/"manifest.json", metadata)
    print(f"Prepared {len(tr)} train / {len(va)} validation / {len(test)} test reviews in {output}")

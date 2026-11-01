"""Validate frozen experiment identity before sharing data or reporting metrics."""
from datasets import load_from_disk
from .common import ROOT, config, digest, read_json, write_json


def main():
    cfg = config()
    data = load_from_disk(str(ROOT / "data/prepared"))
    manifest = read_json(ROOT / "data/manifest.json")
    splits = read_json(ROOT / "data/splits.json")
    assert digest(splits) == manifest["split_hash"], "Split hash changed"
    assert not (set(splits["train"]) & set(splits["validation"])), "Training/validation overlap"
    for split, n in [("train", 20000), ("validation", 5000), ("test", 25000), ("amazon", cfg["amazon_test_size"])]:
        assert len(data[split]) == n, f"Unexpected {split} count"
        assert sum(data[split]["label"]) == n // 2, f"Unbalanced {split}"
        assert len(set(data[split]["id"])) == n, f"Repeated IDs in {split}"
        assert data[split]._fingerprint == manifest["dataset_fingerprints"][split], "Data fingerprint changed"
    for split in ["train", "validation"]:
        assert list(data[split]["id"]) == [f"imdb:train:{i}" for i in splits[split]]
    assert list(data["test"]["id"]) == [f"imdb:test:{i}" for i in range(25000)]
    amazon_ids = read_json(ROOT / "data/amazon_test_indices.json")
    assert digest(amazon_ids) == manifest["amazon_sample_hash"]
    assert list(data["amazon"]["id"]) == [f"amazon:test:{i}" for i in amazon_ids]
    checked = []
    perturbation_manifest = read_json(ROOT / "data/perturbed/manifest.json")
    for kind in ["synonym", "typo", "deletion"]:
        for rate in cfg["perturbation_rates"]:
            name = f"{kind}_{rate:g}"
            ds = load_from_disk(str(ROOT / "data/perturbed" / name))
            assert ds._fingerprint == perturbation_manifest["conditions"][name]["fingerprint"]
            assert list(ds["id"]) == list(data["test"]["id"]), f"ID alignment: {name}"
            assert list(ds["label"]) == list(data["test"]["label"]), f"Label alignment: {name}"
            for row in ds:
                assert row["text"].strip(), f"Empty review: {name}"
                assert row["edit_count"] <= int(row["word_count"] * rate), f"Exceeded budget: {name}"
                assert row["changed"] == (row["edit_count"] > 0)
            checked.append(name)
    result = {"status": "passed", "split_hash": manifest["split_hash"],
              "counts": manifest["counts"], "conditions_checked": checked,
              "semantic_preservation": "Not established by structural verification; review manual_review.csv"}
    write_json(ROOT / "results/data_verification.json", result)
    write_json(ROOT / "data/perturbation_manifest.json", perturbation_manifest)
    print(result)


if __name__ == "__main__":
    main()

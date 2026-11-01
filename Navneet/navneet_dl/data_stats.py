"""Describe evaluation inputs without using their labels to tune the model."""
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from .common import ROOT, read_json, write_json


def main():
    experiment = read_json(ROOT / "models/distilbert/experiment.json")
    max_length = experiment["config"]["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/distilbert")
    data = load_from_disk(str(ROOT / "data/prepared"))
    summary = {}
    for name in ["test", "amazon"]:
        ds = data[name]
        lengths = []
        for start in range(0, len(ds), 512):
            ids = tokenizer(ds[start:start + 512]["text"], truncation=False,
                            padding=False, verbose=False)["input_ids"]
            lengths.extend(map(len, ids))
        summary[name] = {"n": len(ds), "tokens_include_special_tokens": True,
                         "median_tokens": float(np.median(lengths)),
                         "p95_tokens": float(np.percentile(lengths, 95)),
                         "fraction_truncated_at_256": float(np.mean(np.asarray(lengths) > max_length))}
    write_json(ROOT / "results/data_statistics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

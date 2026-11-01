from pathlib import Path
from .io import save_json, write_jsonl
from .engine import train, evaluate, predict


def smoke(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Smoke output already exists")
    positive = ["A wonderful movie with excellent acting", "I loved this warm and moving story",
                "Good performances and a delightful ending", "An enjoyable clever film"]
    negative = ["A terrible movie with awful acting", "I hated this dull and tedious story",
                "Bad performances and a disappointing ending", "A boring dreadful film"]
    for split, repeats in (("train", 8), ("validation", 2), ("test", 2)):
        rows = [{"id": f"synthetic:{split}:{i}:{label}:{j}", "text": text, "label": label}
                for i in range(repeats) for label, texts in ((0, negative), (1, positive)) for j, text in enumerate(texts)]
        write_jsonl(output/"data"/f"{split}.jsonl", rows)
    config = {"seed": 42, "vocab_size": 100, "max_length": 20, "embedding_dim": 8,
              "num_filters": 4, "kernel_sizes": [3, 4, 5], "dropout": .1,
              "batch_size": 16, "epochs": 3, "learning_rate": .01,
              "weight_decay": 0, "patience": 2, "device": "cpu", "threads": 2}
    save_json(output/"config.json", config)
    train(output/"data", output/"model", output/"config.json")
    result, _ = evaluate(output/"model", output/"data/test.jsonl", output/"evaluation", "SYNTHETIC SMOKE ONLY", "cpu")
    prediction = predict(output/"model", "I enjoyed this excellent movie", "cpu")
    save_json(output/"SMOKE_ONLY.json", {"purpose": "Integration test with repeated synthetic examples; not research evidence",
              "completed": True, "prediction": prediction, "metrics": result})

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .common import ROOT, clean, read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--model", default=str(ROOT / "models/distilbert"))
    args = p.parse_args()
    cfg = read_json(f"{args.model}/experiment.json")["config"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    with torch.inference_mode():
        probabilities = model(**tokenizer(clean(args.text), return_tensors="pt", truncation=True,
                              max_length=cfg["max_length"])).logits.softmax(-1)[0]
    label = int(probabilities.argmax())
    print({"sentiment": model.config.id2label[label], "probability": float(probabilities[label])})


if __name__ == "__main__":
    main()

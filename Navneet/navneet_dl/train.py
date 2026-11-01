import argparse
import platform
import subprocess
import sys
import torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from .common import ROOT, config, metrics, read_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Tiny integration run; never report as a project result")
    p.add_argument("--resume", default=None, help="Path to a Trainer checkpoint")
    args = p.parse_args()
    cfg = config()
    set_seed(cfg["seed"])
    name = "smoke" if args.smoke else "distilbert"
    out = ROOT / "runs" / name
    final = ROOT / "models" / name
    if final.exists() and not args.resume:
        raise FileExistsError(f"{final} already exists; preserve it or explicitly resume")
    data = load_from_disk(str(ROOT / "data/prepared"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=2, id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1})
    encoded = {}
    for split in ["train", "validation"]:
        ds = data[split]
        if args.smoke:
            ds = ds.shuffle(seed=42).select(range(32))
        encoded[split] = ds.map(lambda x: tokenizer(x["text"], truncation=True,
                   max_length=cfg["max_length"]), batched=True, remove_columns=["text", "id"])
    training_args = TrainingArguments(
        output_dir=str(out), learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"], max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        weight_decay=cfg["weight_decay"], warmup_ratio=cfg["warmup_ratio"],
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        fp16=torch.cuda.is_available(), seed=cfg["seed"], data_seed=cfg["seed"],
        logging_steps=25, report_to="none", push_to_hub=False, dataloader_num_workers=0,
        disable_tqdm=True)
    def compute(pred):
        result = metrics(pred.label_ids, pred.predictions.argmax(axis=-1))
        return {k: v for k, v in result.items() if k not in {"confusion_matrix", "n"}}
    trainer = Trainer(model=model, args=training_args, train_dataset=encoded["train"],
                      eval_dataset=encoded["validation"], processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute)
    result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    write_json(final / "experiment.json", {
        "config": cfg, "smoke_only": args.smoke,
        "data_manifest": read_json(ROOT / "data/manifest.json"),
        "training_metrics": result.metrics, "validation_metrics": trainer.evaluate(),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "base_model_commit": getattr(model.config, "_commit_hash", None),
        "python": sys.version, "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Offline integration check with a miniature random model, never benchmark evidence."""
import json
import pytest


def test_train_save_reload_and_predict(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from datasets import Dataset, DatasetDict
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, BertTokenizerFast
    from navneet_dl import train
    from navneet_dl.evaluate import predict

    base = tmp_path / "tiny_base"
    base.mkdir()
    (base / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthis\nmovie\nis\ngood\nbad\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(base / "vocab.txt"), do_lower_case=True)
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.save_pretrained(base)
    # AutoTokenizer reload restores model_input_names from the tokenizer class;
    # use DistilBERT tokenizer metadata, whose class omits token_type_ids.
    tc = json.loads((base / "tokenizer_config.json").read_text())
    tc["tokenizer_class"] = "DistilBertTokenizer"
    (base / "tokenizer_config.json").write_text(json.dumps(tc))
    model = DistilBertForSequenceClassification(DistilBertConfig(
        vocab_size=10, max_position_embeddings=32, n_layers=1, n_heads=2,
        dim=16, hidden_dim=32, num_labels=2))
    model.save_pretrained(base)
    records = {"text": ["this movie is good", "this movie is bad"] * 16,
               "label": [1, 0] * 16, "id": [f"fixture:{i}" for i in range(32)]}
    ds = Dataset.from_dict(records)
    DatasetDict(train=ds, validation=ds).save_to_disk(str(tmp_path / "data/prepared"))
    (tmp_path / "data/manifest.json").write_text('{"fixture_only": true}')
    cfg = {"seed": 42, "model": str(base), "max_length": 16, "epochs": 2,
           "learning_rate": 2e-5, "train_batch_size": 8, "eval_batch_size": 16,
           "gradient_accumulation_steps": 1, "weight_decay": .01, "warmup_ratio": .1}
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "config", lambda: cfg)
    monkeypatch.setattr("sys.argv", ["train", "--smoke"])
    train.main()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    saved = tmp_path / "models/smoke"
    assert json.loads((saved / "experiment.json").read_text())["smoke_only"] is True
    loaded = AutoModelForSequenceClassification.from_pretrained(saved)
    tok = AutoTokenizer.from_pretrained(saved)
    predictions, probabilities = predict(loaded, tok, ds, 16, 8)
    assert len(predictions) == len(ds)
    assert set(predictions).issubset({0, 1})
    assert all(0 <= score <= 1 for score in probabilities)

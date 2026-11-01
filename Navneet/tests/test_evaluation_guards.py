import json
import pytest


def test_smoke_checkpoint_cannot_publish_metrics(tmp_path, monkeypatch):
    from navneet_dl import evaluate
    (tmp_path / "experiment.json").write_text(json.dumps({"smoke_only": True}))
    monkeypatch.setattr("sys.argv", ["evaluate", "--model", str(tmp_path)])
    with pytest.raises(ValueError, match="Smoke checkpoints"):
        evaluate.main()


def test_changed_data_manifest_rejected(tmp_path, monkeypatch):
    from navneet_dl import evaluate
    model = tmp_path / "model"
    model.mkdir()
    (model / "experiment.json").write_text(json.dumps({"smoke_only": False, "data_manifest": {"version": 1}}))
    (tmp_path / "data").mkdir()
    (tmp_path / "data/manifest.json").write_text(json.dumps({"version": 2}))
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["evaluate", "--model", str(model)])
    with pytest.raises(ValueError, match="differs"):
        evaluate.main()

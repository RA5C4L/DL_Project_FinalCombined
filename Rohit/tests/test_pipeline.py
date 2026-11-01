import json
import pytest
import torch
from rohit_dl.text import clean_text, tokenize, build_vocab, encode
from rohit_dl.data import split_indices
from rohit_dl.model import TextCNN
from rohit_dl.engine import metrics, train, evaluate, predict
from rohit_dl.perturb import perturb_text
from rohit_dl.io import write_jsonl, save_json, read_jsonl


def test_cleaning_preserves_negation_and_contractions():
    assert clean_text("not<br /> good &amp;  bad") == "not good & bad"
    assert tokenize("I DON'T like it") == ["i", "don't", "like", "it"]


def test_vocab_no_validation_leakage_and_padding():
    vocab = build_vocab(["good movie", "bad movie"], 5)
    assert len(vocab) == 5
    assert encode("unseen", vocab, 5) == [1, 0, 0, 0, 0]
    assert len(encode("good "*1000, vocab)) == 400
    assert vocab == build_vocab(["bad movie", "good movie"], 5)


def test_shared_split_stratified_and_disjoint():
    labels = [0]*12500+[1]*12500
    tr, va = split_indices(labels)
    assert len(tr) == 20000 and len(va) == 5000
    assert not set(tr)&set(va)
    assert set(tr)|set(va) == set(range(25000))
    assert sum(labels[i] for i in va) == 2500
    assert (tr, va) == split_indices(labels)


def test_model_shapes_gradients_padding_invariance():
    torch.manual_seed(42)
    model = TextCNN(20, embedding_dim=8, num_filters=4, dropout=0)
    ids = torch.tensor([[2, 3, 4, 5, 6, 0], [2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    logits = model(ids)
    assert logits.shape == (3, 2) and torch.isfinite(logits).all()
    assert torch.allclose(logits, model(torch.cat([ids, torch.zeros(3, 7, dtype=torch.long)], 1)), atol=1e-6)
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0, 1])).backward()
    assert model.convs[0].weight.grad is not None
    assert torch.equal(model.embedding.weight.grad[0], torch.zeros(8))


def test_metrics_with_missing_predicted_class():
    result = metrics([0, 0, 1, 1], [0, 0, 0, 0])
    assert result["accuracy"] == .5
    assert result["f1_macro"] == pytest.approx(1/3)
    assert result["confusion_matrix"] == [[2, 0], [2, 0]]


@pytest.mark.parametrize("kind", ["synonym", "typo", "deletion"])
def test_corruptions_are_deterministic_and_protect_negation(kind):
    lookup = lambda word: ("excellent",) if word == "great" else ()
    text = "not never great acting and wonderful performances"
    a, stat = perturb_text(text, kind, .5, 42, "x", lookup)
    assert (a, stat) == perturb_text(text, kind, .5, 42, "x", lookup)
    assert "not" in a.split() and "never" in a.split()
    zero, zero_stat = perturb_text(text, kind, 0, 42, "x", lookup)
    assert zero == text and zero_stat["edit_count"] == 0
    assert 0 <= stat["realized_rate"] <= .5


def test_invalid_data_and_rate(tmp_path):
    path = tmp_path/"bad.jsonl"
    write_jsonl(path, [{"id": "1", "text": "x", "label": 2}])
    with pytest.raises(ValueError):
        read_jsonl(path)
    with pytest.raises(ValueError):
        perturb_text("test", "typo", 1.1, 42, "1")


def test_end_to_end_checkpoint_and_overlap_guard(tmp_path):
    from rohit_dl.smoke import smoke
    out = tmp_path/"smoke"
    smoke(out)
    assert json.loads((out/"SMOKE_ONLY.json").read_text())["completed"]
    a = predict(out/"model", "good movie", "cpu")
    b = predict(out/"model", "good movie", "cpu")
    assert a == b
    with pytest.raises(ValueError, match="overlap"):
        evaluate(out/"model", out/"data/train.jsonl", out/"bad-eval", device="cpu")
    with pytest.raises(FileExistsError):
        train(out/"data", out/"model", out/"config.json")


def test_robustness_identity_and_integrity(tmp_path, monkeypatch):
    from rohit_dl import robustness
    from rohit_dl.io import sha256
    clean = tmp_path/'clean.jsonl'
    rows = [{'id': 'test:0', 'text': 'good', 'label': 1}, {'id': 'test:1', 'text': 'bad', 'label': 0}]
    write_jsonl(clean, rows)
    perturbed = tmp_path/'perturbed'
    path = perturbed/'typo_0.1.jsonl'
    write_jsonl(path, [{**r, 'text': 'changed'} for r in rows])
    manifest = {'source_sha256': sha256(clean), 'sets': [{'name': 'typo_0.1', 'path': path.name, 'sha256': sha256(path)}]}
    save_json(perturbed/'manifest.json', manifest)
    def fake_evaluate(run, inp, out, name, device):
        predictions = [{'id': 'test:0', 'label': 1, 'prediction': 1}, {'id': 'test:1', 'label': 0, 'prediction': int(name != 'IMDB clean')}]
        result = metrics([r['label'] for r in predictions], [r['prediction'] for r in predictions])
        out.mkdir(parents=True)
        return result, predictions
    monkeypatch.setattr(robustness, 'evaluate', fake_evaluate)
    summary = robustness.run_suite(tmp_path/'unused-model', clean, perturbed, tmp_path/'suite')
    assert summary[1]['accuracy_drop_pp'] == 50
    assert summary[1]['clean_correct_failure_rate'] == .5
    write_jsonl(path, rows[::-1])
    with pytest.raises(ValueError, match='checksum'):
        robustness.run_suite(tmp_path/'unused-model', clean, perturbed, tmp_path/'suite2')
    manifest['sets'][0]['sha256'] = sha256(path)
    save_json(perturbed/'manifest.json', manifest)
    with pytest.raises(ValueError, match='IDs/labels/order'):
        robustness.run_suite(tmp_path/'unused-model', clean, perturbed, tmp_path/'suite2')


def test_perturbation_manifest_and_rate_edges(tmp_path, monkeypatch):
    import rohit_dl.perturb as perturb
    source = tmp_path/'source.jsonl'
    rows = [{'id': 'test:1', 'text': 'A great movie with wonderful acting', 'label': 1}]
    write_jsonl(source, rows)
    # No WordNet download is needed in CI; exercise all generation and audit paths.
    monkeypatch.setattr(perturb, 'synonyms', lambda word: ())
    monkeypatch.setattr(perturb, 'contextual_tags', lambda text: ())
    original = perturb.perturb_text
    monkeypatch.setattr(perturb, 'perturb_text', lambda text, kind, rate, seed, row_id: original(text, kind, rate, seed, row_id, lambda word: ('excellent',) if word == 'great' else ()))
    manifest = perturb.generate(source, tmp_path/'generated', rates=[0, 1], review_size=1)
    assert len(manifest['sets']) == 6
    assert (tmp_path/'generated/semantic_review.csv').exists()
    zero = read_jsonl(tmp_path/'generated/typo_0.jsonl')[0]
    assert zero['text'] == rows[0]['text']
    deleted = read_jsonl(tmp_path/'generated/deletion_1.jsonl')[0]
    assert deleted['text'].strip()
    with pytest.raises(ValueError):
        perturb.generate(source, tmp_path/'invalid', rates=[.1, .1])


def test_corruption_does_not_edit_function_words_or_acronyms():
    text = 'A film in H.O.T.S. is not without good acting'
    lookup = lambda word: ('changed',)
    for kind in ('synonym', 'deletion', 'typo'):
        altered, _ = perturb_text(text, kind, 1, 42, 'protected', lookup)
        assert 'H.O.T.S.' in altered
        assert altered.startswith('A ')
        assert 'not without' in altered


def test_report_exports_measured_evidence_and_detects_text_leakage(tmp_path):
    from rohit_dl.smoke import smoke
    from rohit_dl.robustness import run_suite
    from rohit_dl.report import export_report
    from rohit_dl.io import sha256, load_json
    root = tmp_path/'report-test'
    smoke(root)
    clean = root/'data/test.jsonl'
    perturbed = root/'perturbed'
    copied = perturbed/'typo_0.jsonl'
    write_jsonl(copied, read_jsonl(clean))
    save_json(perturbed/'manifest.json', {'source_sha256': sha256(clean), 'sets': [
        {'name': 'typo_0', 'path': copied.name, 'sha256': sha256(copied), 'mean_realized_rate': 0, 'unchanged_count': 16}]})
    run_suite(root/'model', clean, perturbed, root/'suite', device='cpu')
    export_report(root/'model', root/'suite', root/'data', root/'report')
    assert (root/'report/RESULTS.md').exists()
    audit = load_json(root/'report/data_audit.json')
    # All smoke test texts were repeated in development. The report must disclose that.
    assert audit['test_excluding_development_text_matches']['n'] == 0
    assert audit['test_excluding_development_text_matches']['accuracy'] is None
    assert (root/'report/training_curves.png').exists()

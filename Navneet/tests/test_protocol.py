import pytest
from navneet_dl.common import clean, metrics
from navneet_dl.perturb import transform
from navneet_dl.prepare import validate_splits


def test_clean_preserves_negation_and_punctuation():
    assert clean("I did NOT<br />like   this!\nNever.") == "I did NOT like this! Never."


def test_metrics_have_fixed_label_order_and_macro_average():
    m = metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["accuracy"] == .75
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["f1_macro"] == pytest.approx((2/3 + .8)/2)


def test_split_overlap_rejected():
    with pytest.raises(ValueError):
        validate_splits({"train": list(range(20000)), "validation": list(range(5000))}, [0]*25000)


@pytest.mark.parametrize("kind", ["synonym", "typo", "deletion"])
def test_zero_rate_exact_identity(kind):
    text = "This was never, ever a boring movie."
    assert transform(text, kind, 0, 42)[0] == text


@pytest.mark.parametrize("kind", ["synonym", "typo", "deletion"])
def test_deterministic_budget_and_negation(kind):
    text = "Never dislike this lovely wonderful movie and beautiful acting."
    lookup = lambda word: ["pleasant"]
    a = transform(text, kind, .4, 42, lookup)
    b = transform(text, kind, .4, 42, lookup)
    assert a == b
    assert len(a[1]) <= int(a[2] * .4)
    assert "Never" in a[0]
    assert all(e["old"] != e["new"] for e in a[1])


def test_synonym_no_candidates_not_falsely_counted():
    text, edits, _ = transform("Lovely acting", "synonym", 1, 42, lambda word: [])
    assert text == "Lovely acting" and edits == []

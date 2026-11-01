"""Deterministic, model-independent stress tests (not targeted adversarial attacks)."""
import csv
import math
import random
import re
from datasets import Dataset, load_from_disk
from .common import ROOT, config, digest, write_json

PROTECTED = {"not", "no", "never", "neither", "nor", "without", "hardly", "barely",
             "don't", "doesn't", "didn't", "isn't", "wasn't", "can't", "won't", "n't"}
TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)


def transform(text, kind, rate, seed, synonym_lookup=None):
    if kind not in {"synonym", "typo", "deletion"}:
        raise ValueError("Unknown perturbation")
    if not 0 <= rate <= 1:
        raise ValueError("Rate must be between zero and one")
    rng = random.Random(seed)
    matches = list(TOKEN.finditer(text))
    words = [i for i, m in enumerate(matches) if m.group().isalpha()]
    eligible = [i for i in words if matches[i].group().lower() not in PROTECTED
                and len(matches[i].group()) >= 4]
    budget = min(len(eligible), math.floor(len(words) * rate))
    edits = []
    rng.shuffle(eligible)
    for i in eligible:
        if len(edits) >= budget:
            break
        m = matches[i]
        original = m.group()
        if kind == "deletion":
            if len(words) - len(edits) <= 1:
                break
            replacement = ""
        elif kind == "typo":
            positions = [j for j in range(1, len(original) - 2) if original[j] != original[j + 1]]
            if not positions:
                continue
            j = rng.choice(positions)
            replacement = original[:j] + original[j + 1] + original[j] + original[j + 2:]
        else:
            choices = sorted(set(synonym_lookup(original.lower()) if synonym_lookup else []))
            choices = [s for s in choices if s.isalpha() and s.lower() != original.lower()
                       and s.lower() not in PROTECTED]
            if not choices:
                continue
            replacement = rng.choice(choices)
            if original.istitle():
                replacement = replacement.capitalize()
            elif original.isupper():
                replacement = replacement.upper()
        edits.append({"start": m.start(), "end": m.end(), "old": original, "new": replacement})
    altered = text
    for e in sorted(edits, key=lambda x: x["start"], reverse=True):
        altered = altered[:e["start"]] + e["new"] + altered[e["end"]:]
    return altered, edits, len(words)


def main():
    from functools import lru_cache
    from nltk.corpus import wordnet
    wordnet.ensure_loaded()  # Install explicitly; never silently omit synonym tests.
    from nlpaug.model.word_dict import WordNet
    synonym_model = WordNet(lang="eng", is_synonym=True)
    @lru_cache(maxsize=50000)
    def synonyms(word):
        return synonym_model.predict(word)
    cfg = config()
    test = load_from_disk(str(ROOT / "data/prepared"))["test"]
    dest = ROOT / "data/perturbed"
    if dest.exists():
        raise FileExistsError("Perturbations already exist; preserve frozen test sets")
    dest.mkdir(parents=True)
    audit = []
    stats = {}
    for kind in ["synonym", "typo", "deletion"]:
        for rate in cfg["perturbation_rates"]:
            name = f"{kind}_{rate:g}"
            records, changed, edits_total, words_total = [], 0, 0, 0
            audit_ids = set(random.Random(cfg["seed"]).sample(range(len(test)), min(50, len(test))))
            for i, row in enumerate(test):
                seed = int(digest([cfg["seed"], row["id"], kind, rate])[:16], 16)
                text, edits, nwords = transform(row["text"], kind, rate, seed, synonyms)
                records.append({"id": row["id"], "text": text, "label": row["label"],
                                "changed": bool(edits), "edit_count": len(edits), "word_count": nwords})
                changed += bool(edits)
                edits_total += len(edits)
                words_total += nwords
                if i in audit_ids:
                    audit.append({"condition": name, "id": row["id"], "label": row["label"],
                                  "original": row["text"], "perturbed": text,
                                  "edits": str(edits), "meaning_preserved": "",
                                  "sentiment_preserved": "", "reviewer_notes": ""})
            frozen = Dataset.from_list(records)
            frozen.save_to_disk(str(dest / name))
            persisted_fingerprint = load_from_disk(str(dest / name))._fingerprint
            stats[name] = {"n": len(test), "changed_reviews": changed,
                           "fingerprint": persisted_fingerprint, "requested_rate": rate,
                           "realized_word_edit_rate": edits_total / max(1, words_total)}
    write_json(dest / "manifest.json", {"seed": cfg["seed"], "conditions": stats,
                "method": "nlpaug WordNet synonym candidates with deterministic custom edit budgets; custom typos/deletion. Not contextual semantic validation"})
    with (dest / "manual_review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)


if __name__ == "__main__":
    main()

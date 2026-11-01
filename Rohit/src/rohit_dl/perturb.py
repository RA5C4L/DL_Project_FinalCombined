"""Reproducible model-independent corruption tests, not optimized adversarial attacks."""
import csv
import hashlib
import random
import re
from functools import lru_cache
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .text import clean_text
from .io import read_jsonl, write_jsonl, save_json, sha256

WORD = re.compile(r"\b[a-zA-Z]+(?:['’][a-zA-Z]+)?\b")
PROTECTED = {"no", "not", "never", "neither", "nor", "without", "hardly", "barely", "scarcely"}


@lru_cache(maxsize=1)
def synonym_model():
    # Use the synonym backend from the library suggested in the project synopsis.
    from nlpaug.model.word_dict import WordNet
    return WordNet(lang="eng", is_synonym=True)


@lru_cache(maxsize=50000)
def synonyms(word, tag="JJ"):
    from nltk.corpus import wordnet as wn
    # First WordNet sense reduces ambiguity but does NOT guarantee contextual equivalence.
    # Restrict to base-form adjectives/adverbs to avoid noun/verb sense and inflection errors.
    pos = {"JJ": wn.ADJ, "RB": wn.ADV}.get(tag)
    if pos is None:
        return ()
    senses = wn.synsets(word, pos=pos)
    if not senses:
        return ()
    first_sense = {lemma.name().lower() for lemma in senses[0].lemmas()}
    candidates = synonym_model().predict(word, pos=pos)
    return tuple(sorted({candidate.lower() for candidate in candidates
        if candidate.lower() in first_sense and candidate.isalpha()
        and candidate.lower() != word and candidate.lower() not in PROTECTED
        and candidate.lower() not in ENGLISH_STOP_WORDS}))


@lru_cache(maxsize=30000)
def contextual_tags(text):
    import nltk
    return tuple(tag for _, tag in nltk.pos_tag([m.group() for m in WORD.finditer(text)]))


def perturb_text(text, kind, rate, seed, row_id, synonym_fn=synonyms):
    if kind not in {"synonym", "typo", "deletion"} or not 0 <= rate <= 1:
        raise ValueError("Unknown perturbation or rate outside [0, 1]")
    text = clean_text(text)
    digest = hashlib.sha256(f"{seed}:{row_id}:{kind}:{rate}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    matches = list(WORD.finditer(text))
    tags = contextual_tags(text) if kind == "synonym" and synonym_fn is synonyms else None
    changes, candidates = {}, []
    for i, match in enumerate(matches):
        token = match.group().lower()
        # Skip acronym components, function words, proper/mixed-case names and negation.
        original = match.group()
        if (len(token) < 4 or token in ENGLISH_STOP_WORDS or token in PROTECTED
                or "'" in token or "’" in token or original.isupper()
                or (any(c.isupper() for c in original) and match.start() != 0)):
            continue
        if kind == "synonym":
            alternatives = synonym_fn(token, tags[i]) if tags is not None else synonym_fn(token)
            if alternatives:
                candidates.append((i, alternatives))
        elif kind == "typo":
            positions = [j for j in range(1, len(token)-2) if token[j] != token[j+1]]
            if positions:
                candidates.append((i, positions))
        else:
            candidates.append((i, None))
    # Rate is requested edits / all word tokens, capped by available eligible tokens.
    target = min(int(len(matches)*rate), len(candidates))
    if kind == "deletion":
        target = min(target, max(0, len(matches)-1))
    for i, options in rng.sample(candidates, target):
        token = matches[i].group()
        if kind == "synonym":
            replacement = rng.choice(options)
            if token.istitle():
                replacement = replacement.capitalize()
        elif kind == "typo":
            j = rng.choice(options)
            replacement = token[:j]+token[j+1]+token[j]+token[j+2:]
        else:
            replacement = ""
        changes[i] = replacement
    result = text
    for i in sorted(changes, reverse=True):
        m = matches[i]
        result = result[:m.start()]+changes[i]+result[m.end():]
    return clean_text(result), {"word_count": len(matches), "eligible_count": len(candidates),
        "edit_count": len(changes), "realized_rate": len(changes)/max(1, len(matches))}


def generate(input_path, output_dir, rates=(.05, .1, .2), seed=42, review_size=30):
    rows, output_dir = read_jsonl(input_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Perturbation directory exists; choose a new directory")
    if not rates or len(set(rates)) != len(rates) or any(not 0 <= r <= 1 for r in rates):
        raise ValueError("Rates must be unique and within [0, 1]")
    # Fail before writing anything if the required corpus is missing.
    synonyms("wonderful")
    contextual_tags("A wonderful movie")
    output_dir.mkdir(parents=True)
    manifest = {"source_sha256": sha256(input_path), "seed": seed, "n": len(rows),
        "method": "nlpaug WordNet backend; contextual base adjective/adverb POS; first sense; internal character swap; random word deletion; protected short/function/name tokens",
        "claim": "Model-independent perturbations; semantic preservation requires human review",
        "sets": []}
    review = []
    for kind in ("synonym", "typo", "deletion"):
        for rate in rates:
            name = f"{kind}_{rate:g}"
            altered, stats = [], []
            for row in rows:
                text, stat = perturb_text(row["text"], kind, rate, seed, row["id"])
                altered.append({**row, "text": text, "perturbation": kind, "requested_rate": rate, **stat})
                stats.append(stat)
            path = output_dir/f"{name}.jsonl"
            write_jsonl(path, altered)
            manifest["sets"].append({"name": name, "path": path.name, "sha256": sha256(path),
                "mean_realized_rate": sum(s["realized_rate"] for s in stats)/len(rows),
                "unchanged_count": sum(s["edit_count"] == 0 for s in stats)})
            sampled = random.Random(seed).sample(range(len(rows)), min(review_size, len(rows)))
            for i in sampled:
                review.append({"condition": name, "id": rows[i]["id"], "label": rows[i]["label"],
                    "original": clean_text(rows[i]["text"]), "perturbed": altered[i]["text"],
                    "sentiment_preserved": "", "grammar_acceptable": "", "reviewer_notes": ""})
            print(f"Generated {name}: {len(rows)} reviews", flush=True)
    with (output_dir/"semantic_review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(review[0]))
        writer.writeheader(); writer.writerows(review)
    save_json(output_dir/"manifest.json", manifest)
    return manifest

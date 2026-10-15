"""Baselines the fine-tuned model has to beat, on the same model input (the serialized telemetry):

  1. tfidf_svm: char 3-5-gram TF-IDF + linear SVM, trained on the train split (all taxonomy classes).
  2. crs_pl1:   OWASP CRS paranoia-level-1 rule regexes (eval.crs_oracle), attack if any rule fires;
                HTTP records only, attack-vs-benign only (CRS has no per-class output).

Scored on test (random lineage groups) and test_unseen (held-out families/variants). A baseline near 100%
on test but not on test_unseen means the split rewards memorised templates.

usage: python -m eval.baselines [--prefix fast_]   -> reports/baselines.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(prefix):
    rows = defaultdict(list)
    for fn in sorted((ROOT / "datasets_ready").glob(f"{prefix}*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            sp = r["metadata"]["split"]
            rows["test_unseen" if sp == "test_unseen" else sp.split("_")[0]].append(
                (r["messages"][1]["content"], r["metadata"]["threat_type"]))
    return rows


def scores(y, p, benign):
    from sklearn.metrics import accuracy_score, f1_score
    yb, pb = [c in benign for c in y], [c in benign for c in p]
    n_benign = sum(yb)
    return {"n": len(y), "accuracy": round(accuracy_score(y, p), 4),
            # macro over the classes present in this split: a class absent from y (e.g. no held-out family)
            # would otherwise enter with F1 = 0 whenever it is predicted, and swing the score with split luck
            "macro_f1": round(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0), 4),
            "attack_vs_benign_accuracy": round(sum(a == b for a, b in zip(yb, pb)) / len(y), 4),
            "benign_fpr": round(sum(t and not q for t, q in zip(yb, pb)) / max(1, n_benign), 4),
            "attack_miss_rate": round(sum((not t) and q for t, q in zip(yb, pb)) / max(1, len(y) - n_benign), 4)}


def tfidf_svm(rows, benign):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=300_000, sublinear_tf=True)
    X = vec.fit_transform([t for t, _ in rows["train"]])
    clf = LinearSVC(C=0.5).fit(X, [c for _, c in rows["train"]])
    out = {}
    for split in ("test", "test_unseen"):
        if rows[split]:
            pred = clf.predict(vec.transform([t for t, _ in rows[split]]))
            out[split] = scores([c for _, c in rows[split]], list(pred), benign)
            per = defaultdict(lambda: [0, 0])
            for (_, c), q in zip(rows[split], pred):
                per[c][0] += c == q
                per[c][1] += 1
            out[split]["per_class_recall"] = {c: round(a / n, 2) for c, (a, n) in sorted(per.items())}
    return out


def crs_pl1(rows, benign):
    from eval.crs_oracle import fields, load_rules
    rules, _ = load_rules()
    pl1 = [(rx, where) for fam in rules.values() for _, rx, where, pl in fam if pl <= 1]
    out = {}
    for split in ("test", "test_unseen"):
        y, p = [], []
        for text, c in rows[split]:
            tel = json.loads(text.split("] ", 1)[1])
            if not tel.get("http"):
                continue
            vals = fields(tel)
            fired = any(rx.search(v) for rx, where in pl1 for w in where for v in vals[w])
            y.append(c in benign)
            p.append(not fired)
        if y:
            n_b = sum(y)
            out[split] = {"n_http": len(y), "attack_vs_benign_accuracy": round(sum(a == b for a, b in zip(y, p)) / len(y), 4),
                          "benign_fpr": round(sum(t and not q for t, q in zip(y, p)) / max(1, n_b), 4),
                          "attack_miss_rate": round(sum((not t) and q for t, q in zip(y, p)) / max(1, len(y) - n_b), 4)}
    return out


def main(prefix):
    benign = {k for k, v in yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"].items() if v["is_benign"]}
    rows = load(prefix)
    rep = {"prefix": prefix, "sizes": {k: len(v) for k, v in rows.items()},
           "tfidf_svm": tfidf_svm(rows, benign), "crs_pl1": crs_pl1(rows, benign),
           "note": "same serialized telemetry the LLM sees; CRS is binary and HTTP-only"}
    (ROOT / "reports/baselines.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: (v if k != "tfidf_svm" else {s: {m: x for m, x in d.items() if m != "per_class_recall"}
                                                      for s, d in v.items()}) for k, v in rep.items()}, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="fast_")
    main(ap.parse_args().prefix)

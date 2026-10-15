"""Shortcut audit (spec §8): can NON-payload surface features predict the label?

Trains a multinomial Naive Bayes (stdlib) on surface tokens only — endpoint, host, header
names/UA, sink position, content type, context keys, flow feature-name set, provenance-free
generator artefacts — on train, scores on val+test. High surface-only accuracy = the dataset
lets a model cheat. Also reports per-feature leakage (train-majority accuracy on held-out).

usage: python -m eval.shortcut_audit [--prefix pilot_]
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def surface(tel):
    """Feature-name -> value, deliberately excluding payload-bearing values."""
    f = {"mode": tel["mode"], "stage": tel["decision_stage"],
         "observed_until": re.sub(r"\d", "#", str(tel.get("observed_until")))}
    h = tel.get("http")
    if h:
        hd = {k.lower(): v for k, v in h["headers"]}
        f["method"] = h["method"]
        f["endpoint"] = re.split(r"[?]", h["target"])[0].split("/")[1] if "/" in h["target"] else "-"
        f["host"] = hd.get("host", "-")
        f["ua"] = re.sub(r"[\d.]+", "#", hd.get("user-agent", "-"))[:40]
        f["header_names"] = ",".join(sorted(hd))
        f["n_headers"] = str(min(len(hd), 12))
        f["ctype"] = hd.get("content-type", "-").split(";")[0]
        f["sinks"] = ",".join(k for k in ("query", "cookies", "body") if h.get(k))
        f["has_views"] = str(bool(h.get("decoded_views")))
        f["protocol"] = h["protocol"]
    if tel.get("flow"):
        f["flow_keys"] = ",".join(sorted(tel["flow"]["features"]))
        f["window"] = tel["flow"]["window"]["kind"]
    f["context_keys"] = ",".join(sorted((tel.get("context") or {}).keys())) or "-"
    f["has_history"] = str(bool(tel.get("history")))
    return f


def structural_tokens(tel):
    """Key paths, header names and enum-like values: things a generator chooses, not payload text."""
    out = {f"observed_until={re.sub(r'[0-9]', '#', str(tel.get('observed_until')))}"}
    h = tel.get("http") or {}
    out |= {f"hdr={k.lower()}" for k, _ in h.get("headers", [])}
    out |= {f"proto={h['protocol']}"} if h else set()
    out |= {f"ctx={k}" for k in (tel.get("context") or {})} | {f"hist={k}" for k in (tel.get("history") or {})}
    if tel.get("flow"):
        out |= {f"feat={k}" for k in tel["flow"]["features"]} | {f"flowproto={tel['flow'].get('protocol')}"}
    return out


def concentrated(recs_tel, min_support=8, benign_floor=0.25):
    """Structural tokens seen only in 1-2 classes WITHOUT a benign counterpart holding >= benign_floor of
    them: candidate generator shortcuts. A class-specific key with matched benign twins is domain
    evidence (GraphQL context means GraphQL traffic), not a shortcut."""
    by = defaultdict(Counter)
    for tel, y in recs_tel:
        for t in structural_tokens(tel):
            by[t][y] += 1
    return {t: dict(c) for t, c in sorted(by.items(), key=lambda kv: -sum(kv[1].values()))
            if sum(c.values()) >= min_support and len(c) <= 2
            and sum(v for k, v in c.items() if k.startswith("benign")) / sum(c.values()) < benign_floor}


def load(prefix):
    recs = []
    for fn in sorted((ROOT / "datasets_ready").glob(f"{prefix}*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            tel = json.loads(r["messages"][1]["content"].split("] ", 1)[1])
            recs.append((surface(tel), r["metadata"]["threat_type"], r["metadata"]["split"], tel))
    return recs


def nb_fit(rows):
    prior, counts, totals, vocab = Counter(), defaultdict(Counter), Counter(), set()
    for feats, y in rows:
        prior[y] += 1
        for k, v in feats.items():
            tok = f"{k}={v}"
            counts[y][tok] += 1
            totals[y] += 1
            vocab.add(tok)
    return prior, counts, totals, len(vocab)


def nb_predict(model, feats):
    prior, counts, totals, V = model
    n = sum(prior.values())
    best, best_lp = None, -math.inf
    for y in prior:
        lp = math.log(prior[y] / n) + sum(
            math.log((counts[y][f"{k}={v}"] + 1) / (totals[y] + V)) for k, v in feats.items())
        if lp > best_lp:
            best, best_lp = y, lp
    return best


def main(prefix):
    import yaml
    benign = {k for k, v in yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"].items() if v["is_benign"]}
    recs = load(prefix)
    train = [(f, y) for f, y, s, _ in recs if s == "train"]
    held = [(f, y) for f, y, s, _ in recs if s != "train"]
    model = nb_fit(train)
    preds = [(nb_predict(model, f), y) for f, y in held]
    acc = sum(p == y for p, y in preds) / len(preds)
    bin_acc = sum((p in benign) == (y in benign) for p, y in preds) / len(preds)
    maj = Counter(y for _, y in train).most_common(1)[0][1] / len(train)
    per_feat = {}
    for k in sorted({k for f, _ in train for k in f}):
        table = defaultdict(Counter)
        for f, y in train:
            table[f.get(k)][y] += 1
        hits = sum(table[f.get(k)].most_common(1)[0][0] == y for f, y in held if table.get(f.get(k)))
        per_feat[k] = round(hits / len(held), 3)
    per_class = defaultdict(lambda: [0, 0])
    for p, y in preds:
        per_class[y][0] += p == y
        per_class[y][1] += 1
    rep = {"prefix": prefix, "n_train": len(train), "n_heldout": len(held),
           "surface_only_nb_accuracy_35class": round(acc, 3),
           "surface_only_nb_accuracy_attack_vs_benign": round(bin_acc, 3),
           "chance_35class": round(1 / 35, 3), "majority_baseline": round(maj, 3),
           "per_feature_majority_accuracy": dict(sorted(per_feat.items(), key=lambda x: -x[1])),
           "per_class_surface_recall": {c: round(a / n, 2) for c, (a, n) in sorted(
               per_class.items(), key=lambda x: -x[1][0] / x[1][1])},
           "concentrated_structural_tokens": concentrated([(t, y) for _, y, _, t in recs]),
           "interpretation": "surface features exclude payload values; accuracy well above chance means "
                             "labels are partly predictable from generator/source artefacts (shortcuts). "
                             "Flow mode is inherently class-informative (L4 classes only exist as flows)."}
    (ROOT / "reports/shortcut_audit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in list(rep)[:7]}, indent=1))
    print("top features:", list(rep["per_feature_majority_accuracy"].items())[:6])
    print("most shortcut-recallable classes:", list(rep["per_class_surface_recall"].items())[:10])
    print("concentrated structural tokens:", len(rep["concentrated_structural_tokens"]))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pilot_")
    main(ap.parse_args().prefix)

"""Blind label review (spec §6 role 5).

export: review packets WITHOUT labels -> data/review/batch_*.jsonl; answer key kept separately in
        data/review_key.json (outside the reviewers' directory).
score:  compare reviewer predictions to proposed labels -> reports/label_audit.json and
        data/review/quarantine.json (ids whose label the reviewer rejected or found unsupported).

Reviewer and generator are the same model family: agreement is a filter, NOT ground truth.
usage: python -m subagents.review export|score
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RV = ROOT / "data" / "review"
KEY = ROOT / "data" / "review_key.json"
N_BATCHES = 3
SEEDS_PER_CLASS = 10


def export():
    from subagents.orchestrator import GAP, OUT
    from data_pipeline.generate import load_seeds
    RV.mkdir(parents=True, exist_ok=True)
    (RV / "out").mkdir(exist_ok=True)
    items, key = [], {}
    rng = random.Random(7)
    for cls in GAP:  # every Haiku-generated record
        for fp in sorted(OUT.glob(f"{cls}.jsonl")) + sorted(OUT.glob(f"{cls}.v*.jsonl")):
            for i, line in enumerate(open(fp)):
                if line.strip():
                    items.append(("telemetry", json.loads(line)["telemetry"], cls, f"{fp.name}:{i}"))
    tax = yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    for cls in tax:  # stratified sample of observed seeds
        seeds = load_seeds(cls)
        for sd in rng.sample(seeds, min(SEEDS_PER_CLASS, len(seeds))):
            items.append(("payload", sd["payload"], cls, sd["id"]))
    rng.shuffle(items)
    batches = defaultdict(list)
    for n, (kind, item, cls, ref) in enumerate(items):
        rid = f"r{n:05d}"
        key[rid] = {"label": cls, "ref": ref, "kind": kind}
        batches[n % N_BATCHES].append({"rid": rid, "kind": kind, "item": item})
    for b, rows in batches.items():
        with open(RV / f"batch_{b}.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    brief = "\n".join(f"- `{k}` ({v['domain']}): {v['scope']}" for k, v in tax.items())
    (RV / "taxonomy_brief.md").write_text(brief + "\n")
    KEY.write_text(json.dumps(key))
    print(len(items), "items in", N_BATCHES, "batches")


def score():
    key = json.loads(KEY.read_text())
    got = {}
    for fp in sorted((RV / "out").glob("batch_*.jsonl")):
        for line in open(fp):
            if line.strip():
                o = json.loads(line)
                got[o["rid"]] = o
    per = defaultdict(Counter)
    quarantine = []
    confusions = Counter()
    for rid, k in key.items():
        o = got.get(rid)
        cls = k["label"]
        if o is None:
            per[cls]["missing"] += 1
            continue
        agree = o.get("predicted_class") == cls
        suff = o.get("evidence_sufficient") is True
        per[cls]["agree" if agree else "disagree"] += 1
        per[cls]["insufficient"] += not suff
        if not agree:
            confusions[(cls, o.get("predicted_class"))] += 1
        if not agree or not suff:
            quarantine.append({"rid": rid, **k, "reviewer": o.get("predicted_class"), "sufficient": suff,
                               "note": o.get("note", "")[:200]})
    rep = {"reviewed": len(got), "items": len(key),
           "caveat": "same-model review; agreement filters noise but is not independent ground truth",
           "per_class": {c: dict(v) | {"agreement": round(v["agree"] / max(1, v["agree"] + v["disagree"]), 2)}
                         for c, v in sorted(per.items())},
           "top_confusions": [[a, b, n] for (a, b), n in confusions.most_common(20)],
           "quarantined": len(quarantine)}
    (ROOT / "reports/label_audit.json").write_text(json.dumps(rep, indent=1))
    (RV / "quarantine.json").write_text(json.dumps(quarantine, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("reviewed", "items", "quarantined")}, indent=1))
    print("lowest agreement:", sorted(((c, v["agreement"]) for c, v in rep["per_class"].items()), key=lambda x: x[1])[:8])
    print("top confusions:", rep["top_confusions"][:8])


def status():
    """ref -> 'agreed' | 'quarantined' for every reviewed item (empty if no review has run)."""
    if not (RV / "quarantine.json").exists():
        return {}
    q = {x["ref"] for x in json.loads((RV / "quarantine.json").read_text())}
    return {k["ref"]: ("quarantined" if k["ref"] in q else "agreed") for k in json.loads(KEY.read_text()).values()}


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1]]()

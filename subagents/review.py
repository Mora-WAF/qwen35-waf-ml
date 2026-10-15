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


def export_crs(per_class=10):
    """Additive review round for CRS fixtures: new rids appended to the existing key; earlier
    batches and their answers stay valid."""
    from data_pipeline.crs import load_pool
    from data_pipeline.build import round_robin_groups
    key = json.loads(KEY.read_text())
    rows = []
    for cls, entries in sorted(load_pool().items()):
        for e in round_robin_groups(entries, per_class):
            rid = f"c{len(key):05d}"
            key[rid] = {"label": cls, "ref": f"crs:{e['original_id']}", "kind": "telemetry"}
            rows.append({"rid": rid, "kind": "telemetry", "item": e["telemetry"]})
    random.Random(11).shuffle(rows)
    with open(RV / "batch_crs.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    KEY.write_text(json.dumps(key))
    print(len(rows), "CRS items -> batch_crs.jsonl")


def export_new(tag):
    """Additive review round for Haiku output lines not yet in the key (new vN files, benign twins).
    Earlier batches and answers stay valid; rids continue the key."""
    from subagents.orchestrator import EXTRA_INGEST, GAP, OUT
    key = json.loads(KEY.read_text())
    seen = {k["ref"] for k in key.values()}
    rows = []
    for cls in GAP + EXTRA_INGEST:
        for fp in sorted(OUT.glob(f"{cls}.jsonl")) + sorted(OUT.glob(f"{cls}.v*.jsonl")):
            for i, line in enumerate(open(fp)):
                ref = f"{fp.name}:{i}"
                if line.strip() and ref not in seen:
                    rid = f"n{len(key):05d}"
                    key[rid] = {"label": cls, "ref": ref, "kind": "telemetry"}
                    rows.append({"rid": rid, "kind": "telemetry", "item": json.loads(line)["telemetry"]})
    random.Random(13).shuffle(rows)
    n_b = max(1, (len(rows) + 119) // 120)  # ~120 items per reviewer batch
    for b in range(n_b):
        with open(RV / f"batch_{tag}_{b}.jsonl", "w") as fh:
            for row in rows[b::n_b]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    KEY.write_text(json.dumps(key))
    print(len(rows), "new items ->", n_b, f"batch_{tag}_*.jsonl")


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
    answered = {json.loads(line)["rid"] for fp in (RV / "out").glob("batch_*.jsonl") for line in open(fp) if line.strip()}
    # an exported item without a reviewer answer is NOT agreed: it stays not_reviewed (needs_review at ingest)
    return {k["ref"]: ("quarantined" if k["ref"] in q else "agreed" if rid in answered else "not_reviewed")
            for rid, k in json.loads(KEY.read_text()).items()}


if __name__ == "__main__":
    if sys.argv[1] == "export_new":
        export_new(sys.argv[2])
    else:
        {"export": export, "export_crs": export_crs, "score": score}[sys.argv[1]]()

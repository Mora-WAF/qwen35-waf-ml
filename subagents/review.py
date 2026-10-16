"""Blind label review (spec §6 role 5).

export: review packets WITHOUT labels -> data/review/batch_*.jsonl; answer key kept separately in
        data/review_key.json (outside the reviewers' directory).
score:  compare reviewer predictions to proposed labels -> reports/label_audit.json and
        data/review/quarantine.json (ids whose label the reviewer rejected or found unsupported).

Reviewer and generator are the same model family: agreement is a filter, NOT ground truth.
usage: python -m subagents.review export|score
"""
import json
import os
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
    write_brief(tax)
    KEY.write_text(json.dumps(key))
    print(len(items), "items in", N_BATCHES, "batches")


def write_brief(tax=None):
    """Reviewer brief: scope plus exclusions (the exclusions carry the boundary rules between classes)."""
    tax = tax or yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    lines = [f"- `{k}` ({v['domain']}): {v['scope']} Not: {'; '.join(v['exclusions'])}." for k, v in tax.items()]
    (RV / "taxonomy_brief.md").write_text("\n".join(lines) + "\n")


def export_recheck(tag, label=None):
    """Second-opinion round for benign<->benign disputes (label benign, reviewer chose another benign class
    with sufficient evidence) under the current brief, or (label=CLASS) for every quarantined item of one class
    after its scope changed. New rids for the same refs: the latest answered round decides (status())."""
    import glob
    tax = yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    write_brief(tax)
    from subagents.orchestrator import OUT
    key = json.loads(KEY.read_text())
    disputed = {x["ref"] for x in json.loads((RV / "quarantine.json").read_text())
                if ((x["label"] == label) if label else
                    (tax.get(x["label"], {}).get("is_benign") and tax.get(x.get("reviewer") or "", {}).get("is_benign")
                     and x.get("sufficient")))}
    rows = []
    for ref in sorted(disputed):
        fname, line = ref.rsplit(":", 1)
        fp = OUT / fname
        if not fp.exists():
            continue
        lines = open(fp).read().split("\n")  # same indexing as `for line in open(fp)`; splitlines() also breaks on U+0085
        label = next(k["label"] for k in key.values() if k["ref"] == ref)
        rid = f"x{len(key):05d}"
        key[rid] = {"label": label, "ref": ref, "kind": "telemetry"}
        rows.append({"rid": rid, "kind": "telemetry", "item": json.loads(lines[int(line)])["telemetry"]})
    random.Random(17).shuffle(rows)
    n_b = max(1, (len(rows) + 119) // 120)
    for b in range(n_b):
        with open(RV / f"batch_{tag}_{b}.jsonl", "w") as fh:
            for row in rows[b::n_b]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    KEY.write_text(json.dumps(key))
    print(len(rows), "disputed benign items ->", n_b, f"batch_{tag}_*.jsonl")


# XM_TAG=<name> keeps a re-audit (e.g. of fixed classes) in its own folder and key, apart from the main audit
_XT = os.environ.get("XM_TAG")
XM = RV / (f"xmodel_{_XT}" if _XT else "xmodel")
XM_KEY = ROOT / "data" / (f"review_xmodel_{_XT}_key.json" if _XT else "review_xmodel_key.json")


def export_xmodel(prefix="fast_", per_class=40, batch=100, classes=None):
    """Cross-family audit sample (spec §6: independent review): per_class records per class from a build,
    labels withheld, reviewed by non-Claude models (Gemini, GPT-OSS via agy). An audit signal: it does not
    quarantine by itself."""
    write_brief()
    by = defaultdict(list)
    for fn in sorted((ROOT / "datasets_ready").glob(f"{prefix}*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            by[r["metadata"]["threat_type"]].append(r)
    rng, key, rows = random.Random(23), {}, []
    for cls in sorted(c for c in by if not classes or c in classes):
        for r in rng.sample(by[cls], min(per_class, len(by[cls]))):
            rid = f"g{len(key):05d}"
            m = r["metadata"]
            key[rid] = {"label": cls, "id": r["id"], "provenance": m["provenance"], "generator": m["generator"],
                        "haiku_review": m["label_evidence"].get("blind_review", "n/a")}
            rows.append({"rid": rid, "item": json.loads(r["messages"][1]["content"].split("] ", 1)[1])})
    rng.shuffle(rows)
    (XM / "out").mkdir(parents=True, exist_ok=True)
    n_b = (len(rows) + batch - 1) // batch
    for b in range(n_b):
        with open(XM / f"batch_{b}.jsonl", "w") as fh:
            for row in rows[b::n_b]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    XM_KEY.write_text(json.dumps(key))
    print(len(rows), "items ->", n_b, "batches in", XM)


def score_xmodel():
    key = json.loads(XM_KEY.read_text())
    tax = yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    got = defaultdict(dict)  # model -> rid -> predicted
    for fp in sorted((XM / "out").glob("*.jsonl")):
        model = fp.name.split("__")[0]
        for line in open(fp):
            try:
                o = json.loads(line)
                got[model][o["rid"]] = o.get("predicted_class")
            except (json.JSONDecodeError, KeyError):
                continue
    models = sorted(got)
    per = {m: defaultdict(lambda: [0, 0]) for m in models}
    binary = {m: [0, 0] for m in models}
    contested = []
    for rid, k in key.items():
        votes = {m: got[m].get(rid) for m in models}
        for m, v in votes.items():
            if v is None:
                continue
            per[m][k["label"]][0] += v == k["label"]
            per[m][k["label"]][1] += 1
            binary[m][0] += tax.get(v, {}).get("is_benign") == tax[k["label"]]["is_benign"]
            binary[m][1] += 1
        if all(v is not None and v != k["label"] for v in votes.values()):  # with one model: its disagreements
            contested.append({"rid": rid, **k, "votes": votes})
    rep = {"models": models, "items": len(key),
           "answered": {m: len(got[m]) for m in models},
           "agreement_35class": {m: round(sum(a for a, _ in per[m].values()) / max(1, sum(n for _, n in per[m].values())), 3) for m in models},
           "agreement_attack_vs_benign": {m: round(binary[m][0] / max(1, binary[m][1]), 3) for m in models},
           "per_class": {c: {m: round(per[m][c][0] / per[m][c][1], 2) for m in models if per[m][c][1]} for c in sorted(tax)},
           "contested_by_all_external": len(contested),
           "contested_by_class": dict(Counter(x["label"] for x in contested).most_common()),
           "contested": contested[:300],
           "caveat": "external models see the same brief; disagreement marks items to inspect, not wrong labels"}
    (ROOT / f"reports/label_audit_xmodel{'_' + _XT if _XT else ''}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("models", "answered", "agreement_35class", "agreement_attack_vs_benign",
                                          "contested_by_all_external", "contested_by_class")}, indent=1))


RELABEL = RV / "relabel.json"


def export_hn(classes=("benign_hard_negative_dev", "benign_hard_negative_search"), batch=100):
    """Every accepted record of the benign hard-negative classes -> blind batches for an external reviewer
    (XM_TAG=hn). The key keeps the raw ref so apply_hn() can relabel at ingest."""
    from subagents.orchestrator import OUT
    write_brief()
    rv, key, rows = status(), {}, []
    for cls in classes:
        for fp in sorted(OUT.glob(f"{cls}.jsonl")) + sorted(OUT.glob(f"{cls}.v*.jsonl")):
            for i, line in enumerate(open(fp)):
                ref = f"{fp.name}:{i}"
                if line.strip() and rv.get(ref) != "quarantined":
                    rid = f"h{len(key):05d}"
                    key[rid] = {"label": cls, "ref": ref}
                    rows.append({"rid": rid, "item": json.loads(line)["telemetry"]})
    random.Random(29).shuffle(rows)
    (XM / "out").mkdir(parents=True, exist_ok=True)
    n_b = (len(rows) + batch - 1) // batch
    for b in range(n_b):
        with open(XM / f"batch_{b}.jsonl", "w") as fh:
            for row in rows[b::n_b]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    XM_KEY.write_text(json.dumps(key))
    print(len(rows), "hard-negative items ->", n_b, "batches in", XM)


def apply_hn():
    """External verdicts on hard negatives -> relabel.json: another benign class = relabel (benign<->benign
    moves cannot create false blocks); an attack class = quarantine (never auto-promote benign to attack)."""
    import yaml as _y
    tax = _y.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    key = json.loads(XM_KEY.read_text())
    got = {}
    for fp in (XM / "out").glob("*.jsonl"):
        for line in open(fp):
            try:
                o = json.loads(line)
                got[o["rid"]] = o.get("predicted_class")
            except (json.JSONDecodeError, KeyError):
                continue
    rel = json.loads(RELABEL.read_text()) if RELABEL.exists() else {}
    moved, quar = Counter(), Counter()
    for rid, k in key.items():
        p = got.get(rid)
        if p is None or p == k["label"] or p not in tax:
            continue
        if tax[p]["is_benign"]:
            rel[k["ref"]] = p
            moved[(k["label"], p)] += 1
        else:
            rel[k["ref"]] = "__quarantine__"
            quar[(k["label"], p)] += 1
    RELABEL.write_text(json.dumps(rel, indent=1))
    print(json.dumps({"answered": len(got), "of": len(key), "relabelled": {f"{a}->{b}": n for (a, b), n in moved.items()},
                      "quarantined": {f"{a}->{b}": n for (a, b), n in quar.items()}}, indent=1))


def relabels():
    return json.loads(RELABEL.read_text()) if RELABEL.exists() else {}


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
    q = {x["rid"] for x in json.loads((RV / "quarantine.json").read_text())}
    answered = {json.loads(line)["rid"] for fp in (RV / "out").glob("batch_*.jsonl") for line in open(fp) if line.strip()}
    # key order = export order: the latest ANSWERED round for a ref decides (recheck rounds can clear or
    # confirm a quarantine); an exported item without any answer stays not_reviewed (needs_review at ingest)
    out = {}
    for rid, k in json.loads(KEY.read_text()).items():
        if rid in answered:
            out[k["ref"]] = "quarantined" if rid in q else "agreed"
        else:
            out.setdefault(k["ref"], "not_reviewed")
    return out


if __name__ == "__main__":
    if sys.argv[1] in ("export_hn", "apply_hn"):
        {"export_hn": export_hn, "apply_hn": apply_hn}[sys.argv[1]]()
    elif sys.argv[1] == "export_xmodel":  # export_xmodel [PREFIX] [CLASS,CLASS...] [PER_CLASS]
        export_xmodel(*(sys.argv[2:3] or ["fast_"]), classes=set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None,
                      **({"per_class": int(sys.argv[4])} if len(sys.argv) > 4 else {}))
    elif sys.argv[1] == "score_xmodel":
        score_xmodel()
    elif sys.argv[1] in ("export_new", "export_recheck"):
        {"export_new": export_new, "export_recheck": export_recheck}[sys.argv[1]](*sys.argv[2:4])  # TAG [LABEL]
    else:
        {"export": export, "export_crs": export_crs, "score": score}[sys.argv[1]]()

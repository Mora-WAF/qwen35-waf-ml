"""Source-matched splits for single-campaign observed flows (post-processing of a finished build).

Every CIC-IDS2017 attack flow shares one lineage group (`cic:172.16.0.1->192.168.10.50`: one attacker host, all
classes), so the build puts the whole campaign in one split. In the 18 Oct fast build that was val_select, while 941
CIC-IDS2017 benign flows sat in train: the CIC-2017 feature-set shape meant "benign" in training, and Gemma called
1,155 of those attacks benign (port scan, credential stuffing, slowloris).

This orders each affected class by source row id (capture order). The latest `--holdout` share stays where the build
put it (a later-traffic test of the same campaign: weaker than an unseen source); the earlier rest moves to train.
The two parts get their own lineage groups (`#early` / `#late`), so the lineage checks still hold. Every other record
keeps its split, so models trained on the result compare cleanly with ones trained on the source build.

Since 19 Oct the build applies the same rule itself (data_pipeline.build -> block_campaigns, every observed source):
a build no longer needs this post-processing step; the CLI stays for reproducing datasets_cicfix/.

usage: .venv/bin/python -m data_pipeline.block_split SRC_DIR PREFIX OUT_DIR [--holdout 0.2]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test", "test_unseen")


def _capture_order(r):
    o = str(r["metadata"]["original_id"])
    return (0, int(o), "") if o.isdigit() else (1, 0, o)


def block_campaigns(records, holdout=0.2, sources=None):
    """In place on build records (metadata split / lineage_group): every observed attack class of a source that has
    no observed record of that source in train, and is not held out as unseen, is ordered by capture (source row id);
    the earlier 1 - holdout moves to train (`#early`), the latest stay in their split (`#late`). sources: None = all.
    Returns {"source/class": {to_train, held_out, held_out_split}}."""
    cand = defaultdict(list)
    for r in records:
        m = r["metadata"]
        if m["provenance"] == "observed" and not m["threat_type"].startswith("benign") and (sources is None or m["source"] in sources):
            cand[(m["source"], m["threat_type"])].append(r)
    moved = {}
    for (src, cls), rs in sorted(cand.items()):
        splits = {r["metadata"]["split"] for r in rs}
        if "train" in splits or "test_unseen" in splits or len(rs) < 2:  # unseen sources stay unseen
            continue
        rs.sort(key=_capture_order)
        n_keep = max(1, round(len(rs) * holdout))
        for i, r in enumerate(rs):
            m = r["metadata"]
            early = i < len(rs) - n_keep  # earlier capture -> train; the latest n_keep stay held out
            m["lineage_group"] = f"{m['lineage_group']}#{cls}#{'early' if early else 'late'}"
            if early:
                m["split"] = "train"
        moved[f"{src}/{cls}"] = {"to_train": len(rs) - n_keep, "held_out": n_keep, "held_out_split": sorted(splits)}
    return moved


def align_unseen_sources(records):
    """A source whose observed attack flows are all held out as unseen (test_unseen) must not keep its benign flows in
    train either: training would see only that source's benign shape (the CIC-2017 bug, by another route). Their whole
    lineage groups move to test_unseen, so the source is unseen on both sides. Returns {source: records moved}."""
    att = defaultdict(set)
    for r in records:
        m = r["metadata"]
        if m["provenance"] == "observed" and not m["threat_type"].startswith("benign"):
            att[m["source"]].add(m["split"])
    unseen_only = {s for s, sp in att.items() if sp == {"test_unseen"}}
    groups = {r["metadata"]["lineage_group"] for r in records if r["metadata"]["provenance"] == "observed"
              and r["metadata"]["source"] in unseen_only and r["metadata"]["split"] != "test_unseen"}
    moved = defaultdict(int)
    for r in records:
        if r["metadata"]["lineage_group"] in groups:
            r["metadata"]["split"] = "test_unseen"
            moved[r["metadata"]["source"]] += 1
    return dict(moved)


def observed_per_split(records):
    """{source: {split: {"attack": n, "benign": n}}} for observed flows: a source whose attacks are missing from a split
    that holds its benign flows teaches 'this source's feature-set shape = benign' there (the 18 Oct CIC-2017 bug)."""
    out = defaultdict(lambda: defaultdict(lambda: {"attack": 0, "benign": 0}))
    for r in records:
        m = r["metadata"]
        if m["provenance"] == "observed":
            out[m["source"]][m["split"]]["benign" if m["threat_type"].startswith("benign") else "attack"] += 1
    return {s: dict(v) for s, v in out.items()}


def main(src, prefix, out, holdout=0.2, source="cic_ids2017_mirror"):
    src, out = ROOT / src, ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    recs = {s: [json.loads(x) for x in open(src / f"{prefix}{s}.jsonl")] for s in SPLITS}
    before = {r["id"]: s for s in SPLITS for r in recs[s]}
    moved = {k.split("/", 1)[1]: v for k, v in block_campaigns([r for s in SPLITS for r in recs[s]], holdout, {source}).items()}
    to_train = {r["id"] for s in SPLITS[1:] for r in recs[s] if r["metadata"]["split"] == "train" and before[r["id"]] != "train"}
    for s in SPLITS[1:]:
        recs["train"] += [r for r in recs[s] if r["id"] in to_train]
        recs[s] = [r for r in recs[s] if r["id"] not in to_train]
    for s in SPLITS:
        with open(out / f"{prefix}{s}.jsonl", "w") as f:
            for r in recs[s]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # every group must sit in one split (block groups included)
    where = defaultdict(set)
    for s in SPLITS:
        for r in recs[s]:
            where[r["metadata"]["lineage_group"]].add("val" if r["metadata"]["split"].startswith("val") else s)
    bad = [g for g, ss in where.items() if len(ss) > 1]
    assert not bad, f"lineage groups across splits: {bad[:5]}"
    rep = {"source_build": str(src.relative_to(ROOT)), "prefix": prefix, "holdout": holdout, "moved": moved,
           "counts": {s: len(recs[s]) for s in SPLITS}}
    (out / "block_split.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("prefix")
    ap.add_argument("out")
    ap.add_argument("--holdout", type=float, default=0.2)
    a = ap.parse_args()
    main(a.src, a.prefix, a.out, a.holdout)

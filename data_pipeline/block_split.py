"""Source-matched splits for single-campaign observed flows (post-processing of a finished build).

Every CIC-IDS2017 attack flow shares one lineage group (`cic:172.16.0.1->192.168.10.50`: one attacker host, all
classes), so the build puts the whole campaign in one split. In the 18 Oct fast build that was val_select, while 941
CIC-IDS2017 benign flows sat in train: the CIC-2017 feature-set shape meant "benign" in training, and Gemma called
1,155 of those attacks benign (port scan, credential stuffing, slowloris).

This orders each affected class by source row id (capture order). The latest `--holdout` share stays where the build
put it (a later-traffic test of the same campaign: weaker than an unseen source); the earlier rest moves to train.
The two parts get their own lineage groups (`#early` / `#late`), so the lineage checks still hold. Every other record
keeps its split, so models trained on the result compare cleanly with ones trained on the source build.

usage: .venv/bin/python -m data_pipeline.block_split SRC_DIR PREFIX OUT_DIR [--holdout 0.2]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test", "test_unseen")


def main(src, prefix, out, holdout=0.2, source="cic_ids2017_mirror"):
    src, out = ROOT / src, ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    recs = {s: [json.loads(x) for x in open(src / f"{prefix}{s}.jsonl")] for s in SPLITS}
    # attack classes of this source with no observed record of it in train: the campaign sits wholly elsewhere
    cand = defaultdict(list)
    for s in SPLITS:
        for r in recs[s]:
            m = r["metadata"]
            if m["source"] == source and m["provenance"] == "observed" and m["threat_type"] != "benign_standard_traffic":
                cand[m["threat_type"]].append((s, r))
    moved, to_train = {}, set()
    for cls, rs in sorted(cand.items()):
        if any(s == "train" for s, _ in rs):
            continue
        rs.sort(key=lambda x: int(x[1]["metadata"]["original_id"]))
        n_keep = max(1, round(len(rs) * holdout))
        for i, (s, r) in enumerate(rs):
            m = r["metadata"]
            early = i < len(rs) - n_keep  # earlier capture -> train; the latest n_keep stay held out
            m["lineage_group"] = f"{m['lineage_group']}#{cls}#{'early' if early else 'late'}"
            if early:
                m["split"] = "train"
                to_train.add(r["id"])
        moved[cls] = {"to_train": len(rs) - n_keep, "held_out": n_keep, "held_out_split": sorted({s for s, _ in rs})}
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

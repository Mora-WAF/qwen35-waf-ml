"""Assemble a dataset build from seeds + flow/benign oracles + Haiku scenario outputs (spec §B/§C).

Pilot mode (default): ~`per_class` records/class, enough to exercise every path end to end.
Writes datasets_ready/*.jsonl + manifest.json + dataset_stats_report.json.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets_ready"


def taxonomy():
    return yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]


def round_robin_groups(entries, k):
    """Take k entries spreading over lineage groups before repeating any group."""
    groups = {}
    for e in entries:
        groups.setdefault(e["lineage_group"], []).append(e)
    out, i = [], 0
    while len(out) < k and any(len(g) > i for g in groups.values()):
        out += [g[i] for g in groups.values() if len(g) > i][: k - len(out)]
        i += 1
    return out


def collect(per_class, frac=None):
    """per_class records per class, or (frac) ceil(frac x quota) per class for a quota-proportional build."""
    from data_pipeline import generate as G
    from subagents.orchestrator import ingest as ingest_haiku, GAP, GAP_BENIGN
    tax = taxonomy()
    want_of = (lambda c: -(-tax[c]["quota"] * frac // 1)) if frac else (lambda c: per_class)
    recs = []
    deficits = {}
    # Haiku scenario classes
    haiku_recs, haiku_report = ingest_haiku()
    haiku_by_cls = Counter(r["metadata"]["threat_type"] for r in haiku_recs)
    recs += haiku_recs
    # seeded L7 classes
    from data_pipeline.crs import load_pool as load_crs
    crs_pool = load_crs()
    from subagents.review import status
    rv = status()
    seeded = [c for c in tax if (G.SEEDS / f"{c}.jsonl").exists() and c not in GAP]
    for cls in seeded:
        seeds = G.load_seeds(cls)
        if not seeds:
            continue
        # a quarter from CRS rule fixtures (round-robin over rules) where the class has them
        # CRS rule fixtures are regex unit tests: 35% of a blind-reviewed sample lacked evidence, so only
        # fixtures the review AGREED with are used (unreviewed ones carry the same noise rate)
        crs = [e for e in crs_pool.get(cls, []) if rv.get(f"crs:{e['original_id']}") == "agreed"]
        picks = round_robin_groups(crs, int(want_of(cls)) // 4) if crs else []
        recs += G.gen_observed_http(picks)
        want = int(want_of(cls)) - len(picks)
        # spread over as many distinct seeds as possible before re-wrapping any seed
        base, extra = divmod(want, len(seeds))
        for j, s in enumerate(seeds):
            n = base + (j < extra)
            if n:
                recs += G.gen_from_seed(s, n)
    # flows: observed first (round-robin over campaigns, across sources); synthetic only where no
    # observed source exists, or as a flagged top-up when observed campaigns < 5 (cannot reach val/test)
    from data_pipeline.flows import load_pool
    pool = load_pool()
    mixed = {}
    for cls in G.FLOW_PROFILES:
        # benign flows: real NIDS traffic is mostly benign; 2x per_class so "flow" alone does not imply attack
        want = int(want_of(cls))
        if cls == "benign_standard_traffic":
            want = 2 * per_class if not frac else want // 3  # flow share of benign_standard at scale
        obs = pool.get(cls, [])
        groups = {}
        for e in obs:
            groups.setdefault(e["lineage_group"], []).append(e)
        picks, i = [], 0
        while len(picks) < want and any(len(g) > i for g in groups.values()):
            picks += [g[i] for g in groups.values() if len(g) > i][: want - len(picks)]
            i += 1
        if obs:  # always half observed / half synthetic: counterbalances source feature-set shortcuts
            picks = picks[: want // 2]
            mixed[cls] = {"observed_groups": len(groups), "observed": len(picks), "synthetic": want - len(picks)}
        recs += G.gen_observed_flow(picks)
        if len(picks) < want:
            recs += G.gen_flow(cls, want - len(picks))
    # HTTP part of benign_standard (twins and Haiku counterparts add more)
    recs += G.gen_benign_std("benign_standard_traffic", per_class // 2 if not frac else int(want_of("benign_standard_traffic")) // 3)
    # record deficits vs quota-intent for pilot
    have = Counter(r["metadata"]["threat_type"] for r in recs)
    for cls in tax:
        if have[cls] < want_of(cls):
            deficits[cls] = {"have": have[cls], "target": int(want_of(cls)),
                             "reason": "haiku job empty/missing" if cls in GAP and haiku_by_cls[cls] == 0
                                       else "insufficient independent seeds"}
    deficits["_mixed_observed_synthetic"] = mixed
    return recs, deficits, haiku_report


def enforce_budget(recs):
    """Reject records whose full rendered sequence exceeds the token budget (spec §7). No silent
    truncation: rejected ids are reported; token_len written into metadata for the rest."""
    from training.prepare import load_tokenizer, tokenize_with_mask, MAX_TOKENS
    tok, kept, over = load_tokenizer(), [], Counter()
    for r in recs:
        n = len(tokenize_with_mask(tok, r["messages"])[0])
        if n > MAX_TOKENS:
            over[r["metadata"]["threat_type"]] += 1
            continue
        r["metadata"]["token_len"] = n
        kept.append(r)
    return kept, dict(over)


def build(per_class=24, ratios=(0.8, 0.1, 0.1), frac=None, prefix="pilot_"):
    from data_pipeline.dedup_and_split import (dedup, assign_splits, split_counts, holdout_unseen,
                                               cross_split_overlap, lineage_overlap)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").unlink(missing_ok=True)  # a crashed build must never leave a gate-passable manifest
    recs, deficits, haiku_report = collect(per_class, frac)
    recs, dropped = dedup(recs)
    recs, overflow = enforce_budget(recs)
    unseen = holdout_unseen(recs)
    recs = assign_splits([r for r in recs if r["metadata"]["lineage_group"] not in unseen], ratios) + \
        [r for r in recs if r["metadata"]["lineage_group"] in unseen]
    for r in recs:
        if r["metadata"]["lineage_group"] in unseen:
            r["metadata"]["split"] = "test_unseen"
    xo, lo = cross_split_overlap(recs), lineage_overlap(recs)
    assert not lo, f"lineage overlap across splits: {list(lo)[:5]}"
    assert not xo, f"content overlap across splits: {len(xo)} hashes"

    by_split = {"train": [], "val": [], "test": [], "test_unseen": []}
    for r in recs:
        sp = r["metadata"]["split"]
        by_split[sp if sp == "test_unseen" else sp.split("_")[0]].append(r)
    files = {k: f"{prefix}{k}.jsonl" for k in by_split}
    manifest = {"mode": prefix.rstrip("_"), "per_class": per_class, "quota_fraction": frac, "total": len(recs), "dropped_exact_dup": dropped,
                "rejected_token_overflow": overflow,
                "files": {}, "deficits": deficits, "haiku_jobs": haiku_report,
                "cross_split_overlap": len(xo), "lineage_overlap": len(lo),
                "limitations": ["CIC-IDS2017 attack classes are single campaigns (one lineage group each); "
                                "they land wholly in one split, test coverage for those classes comes from NF-ToN or synthetic",
                                "synthetic flow/benign/haiku records are independent draws (own lineage group) but share per-class template families across splits; no unseen-family holdout in pilot",
                                "near-duplicate (non-exact) seed variants not yet clustered"]}
    for split, fname in files.items():
        p = OUT / fname
        with open(p, "w") as f:
            for r in by_split[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["files"][fname] = {"records": len(by_split[split]),
                                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    stats = {"split_counts": split_counts(recs),
             "provenance": dict(Counter(r["metadata"]["provenance"] for r in recs)),
             "classes_present": sorted({r["metadata"]["threat_type"] for r in recs})}
    (OUT / "dataset_stats_report.json").write_text(json.dumps(stats, indent=1))
    return manifest, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--quota-frac", type=float, default=None, help="e.g. 0.25 -> ~30k quota-proportional build")
    ap.add_argument("--prefix", default="pilot_")
    a = ap.parse_args()
    m, s = build(a.per_class, frac=a.quota_frac, prefix=a.prefix)
    print(json.dumps({"total": m["total"], "dropped": m["dropped_exact_dup"],
                      "classes_present": len(s["classes_present"]),
                      "deficits": {k: v["have"] for k, v in m["deficits"].items() if not k.startswith("_")},
                      "mixed_observed_synthetic": m["deficits"].get("_mixed_observed_synthetic")}, indent=1))

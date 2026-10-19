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


# classes whose label is defined by repetition across connections (a cross-model audit rejected single
# observed flows of this class without window evidence: 15 of 17)
NEEDS_WINDOW = {"credential_stuffing_bruteforce", "port_scan_recon"}  # scans: distinct ports, also one-flow-blind


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
    src_attack = Counter()  # observed attack flows taken per source
    # attack classes first; benign_standard last, matched per source (see below)
    for cls in sorted(G.FLOW_PROFILES, key=lambda c: c == "benign_standard_traffic"):
        # benign flows: real NIDS traffic is mostly benign; 2x per_class so "flow" alone does not imply attack
        want = int(want_of(cls))
        if cls == "benign_standard_traffic":
            want = 2 * per_class if not frac else want // 3  # flow share of benign_standard at scale
        obs = pool.get(cls, [])
        if cls in NEEDS_WINDOW:  # one flow cannot show many attempts: observed rows need the window count
            obs = [e for e in obs if "conn_count_60s" in e["telemetry"]["flow"]["features"]]  # CIC only (NF has no timestamps)
        groups = {}
        for e in obs:
            groups.setdefault(e["lineage_group"], []).append(e)
        if cls == "benign_standard_traffic":
            # observed benign flows 1:1 with each source's observed attack flows: a source's feature-set shape
            # (no bytes in DNS-EXF, no window counts in IDS2018, CIC field quirks) must not predict the label;
            # a source with no observed attacks (NF-ToN after the window rule) contributes no benign either
            picks = []
            for src, n in sorted(src_attack.items()):
                sg = {g: v for g, v in groups.items() if v[0]["source"] == src}
                take, i = [], 0
                while len(take) < n and any(len(g) > i for g in sg.values()):
                    take += [g[i] for g in sg.values() if len(g) > i][: n - len(take)]
                    i += 1
                picks += take
            mixed[cls] = {"observed_groups": len(groups), "observed": len(picks), "synthetic": want,
                          "matched_per_source": {k: sum(p["source"] == k for p in picks) for k in src_attack}}
            recs += G.gen_observed_flow(picks)
            recs += G.gen_flow(cls, want)
            continue
        # round-robin over sources, then lineage groups, so one large source cannot crowd out the others
        by_src = {}
        for g, v in groups.items():
            by_src.setdefault(v[0]["source"], {})[g] = v
        picks, i = [], 0
        cap = want // 2  # always half observed / half synthetic: counterbalances source feature-set shortcuts
        order = [list(sg.values()) for _, sg in sorted(by_src.items())]
        while len(picks) < cap and any(len(g) > i for sg in order for g in sg):
            for sg in order:
                picks += [g[i] for g in sg if len(g) > i][: cap - len(picks)]
            i += 1
        if obs:
            mixed[cls] = {"observed_groups": len(groups), "observed": len(picks), "synthetic": want - len(picks),
                          "per_source": {k: sum(p["source"] == k for p in picks) for k in by_src}}
            for p in picks:
                src_attack[p["source"]] += 1
        recs += G.gen_observed_flow(picks)
        if len(picks) < want:
            recs += G.gen_flow(cls, want - len(picks))
    # HTTP part of benign_standard (twins and Haiku counterparts add more)
    recs += G.gen_benign_std("benign_standard_traffic", per_class // 2 if not frac else int(want_of("benign_standard_traffic")) // 3)
    recs += G.gen_fragments(300 if frac else per_class)  # insufficient-context fragments: benign by spec
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
    from data_pipeline.dedup_and_split import (dedup, drop_label_conflicts, assign_splits, split_counts, holdout_unseen,
                                               cross_split_overlap, lineage_overlap)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").unlink(missing_ok=True)  # a crashed build must never leave a gate-passable manifest
    recs, deficits, haiku_report = collect(per_class, frac)
    recs, label_conflicts = drop_label_conflicts(recs)  # same input, two labels: neither copy is trained on
    recs, dropped = dedup(recs)
    recs, overflow = enforce_budget(recs)
    unseen = holdout_unseen(recs)
    recs = assign_splits([r for r in recs if r["metadata"]["lineage_group"] not in unseen], ratios) + \
        [r for r in recs if r["metadata"]["lineage_group"] in unseen]
    for r in recs:
        if r["metadata"]["lineage_group"] in unseen:
            r["metadata"]["split"] = "test_unseen"
    # single-campaign observed classes (every CIC-IDS2017 attack is one attacker host = one lineage group) land wholly
    # outside train while the source's benign flows are in it: the source's feature-set shape then means benign
    # (18-19 Oct: real port scans 0/500 and brute force 0/470 allowed). Earlier capture -> train, the latest 20% held out.
    from data_pipeline.block_split import align_unseen_sources, block_campaigns, observed_per_split
    blocked = block_campaigns(recs)
    aligned = align_unseen_sources(recs)  # a source held out as unseen leaves train with its benign flows too
    per_split = observed_per_split(recs)
    shape_only = [src for src, sp in per_split.items() if sp.get("train", {}).get("benign") and not sp.get("train", {}).get("attack")
                  and any(v["attack"] for v in sp.values())]
    assert not shape_only, f"observed sources with benign but no attack flows in train: {shape_only}"
    xo, lo = cross_split_overlap(recs), lineage_overlap(recs)
    assert not lo, f"lineage overlap across splits: {list(lo)[:5]}"
    assert not xo, f"content overlap across splits: {len(xo)} hashes"

    by_split = {"train": [], "val": [], "test": [], "test_unseen": []}
    for r in recs:
        sp = r["metadata"]["split"]
        by_split[sp if sp == "test_unseen" else sp.split("_")[0]].append(r)
    files = {k: f"{prefix}{k}.jsonl" for k in by_split}
    manifest = {"mode": prefix.rstrip("_"), "per_class": per_class, "quota_fraction": frac, "total": len(recs), "dropped_exact_dup": dropped,
                "dropped_label_conflicts": label_conflicts,
                "rejected_token_overflow": overflow,
                "files": {}, "deficits": deficits, "haiku_jobs": haiku_report,
                "cross_split_overlap": len(xo), "lineage_overlap": len(lo),
                "observed_campaign_blocks": blocked, "unseen_source_benign_moved": aligned, "observed_per_split": per_split,
                "limitations": ["CIC-IDS2017 attack classes are single campaigns (one attacker host): the build trains on "
                                "the earlier 80% of each (capture order) and holds out the latest 20% (observed_campaign_blocks), "
                                "a later-traffic test of the same campaign, weaker than an unseen source",
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

"""Grouping, dedup, and leakage-safe split assignment (spec §8).

Order matters: group by lineage FIRST, then allocate whole groups to splits, so no seed's
descendants straddle a split. Exact content hashing catches duplicates across sources. We
report "zero detected cross-split overlap under the documented checks" — never zero semantic
leakage.
"""
import hashlib
import json
import random
from collections import defaultdict

SPLITS = ("train", "val", "test")
# spec §8: val = 50% model selection, 25% calibrator fit, 25% policy thresholds; disjoint by group
VAL_PARTS = ("val_select", "val_calib", "val_policy", "val_select")


def content_hash(rec):
    return hashlib.sha256(rec["messages"][1]["content"].encode()).hexdigest()


def dedup(records):
    """Drop exact user-content duplicates, keeping the first. Returns (kept, n_dropped)."""
    seen, kept = set(), []
    dropped = 0
    for r in records:
        h = content_hash(r)
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        r["metadata"]["dup_cluster"] = h[:16]
        kept.append(r)
    return kept, dropped


def assign_splits(records, ratios=(0.8, 0.1, 0.1), seed=1337):
    """Assign whole lineage groups to train/test/val_* GLOBALLY: one group -> one split, even when
    the group spans classes (e.g. one attacker host doing scanning AND brute force). Classes with
    the fewest groups pick first so rare classes still reach val and test."""
    by_class_groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_class_groups[r["metadata"]["threat_type"]][r["metadata"]["lineage_group"]].append(r)
    rng = random.Random(seed)
    assigned = {}
    for cls in sorted(by_class_groups, key=lambda c: (len(by_class_groups[c]), c)):
        gids = sorted(by_class_groups[cls])
        rng.shuffle(gids)
        n = len(gids)
        # >=1 test group when >=2 groups; >=3 val groups (one per subpartition) when >=5 groups
        n_test = max(1, int(n * ratios[2])) if n >= 3 else (1 if n >= 2 else 0)
        n_val = max(3, int(n * ratios[1])) if n >= 5 else 0
        have_test = sum(assigned.get(g) == "test" for g in gids)
        have_val = sum(assigned.get(g, "").startswith("val") for g in gids)
        for gid in gids:
            if gid in assigned:
                continue
            if have_test < n_test:
                assigned[gid], have_test = "test", have_test + 1
            elif have_val < n_val:  # round-robin -> ~2:1:1 select/calib/policy
                assigned[gid], have_val = VAL_PARTS[have_val % len(VAL_PARTS)], have_val + 1
            else:
                assigned[gid] = "train"
    for r in records:
        r["metadata"]["split"] = assigned[r["metadata"]["lineage_group"]]
    return records


def split_counts(records):
    c = defaultdict(lambda: defaultdict(int))
    for r in records:
        c[r["metadata"]["threat_type"]][r["metadata"]["split"]] += 1
    return {k: dict(v) for k, v in c.items()}


def cross_split_overlap(records):
    """Return content hashes that appear in >1 split (should be empty after group assignment)."""
    by_hash = defaultdict(set)
    for r in records:
        by_hash[content_hash(r)].add(r["metadata"]["split"])
    return {h: sorted(s) for h, s in by_hash.items() if len(s) > 1}


def lineage_overlap(records):
    """Return lineage groups whose members span >1 split (a leakage bug)."""
    by_grp = defaultdict(set)
    for r in records:
        by_grp[r["metadata"]["lineage_group"]].add(r["metadata"]["split"])
    return {g: sorted(s) for g, s in by_grp.items() if len(s) > 1}

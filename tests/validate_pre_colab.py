"""Pre-Colab gate (spec §12). Exits non-zero on any failure; writes reports/pre_colab_gate.json.

usage: python -m tests.validate_pre_colab [--prefix pilot_]
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training import prepare as P  # noqa: E402

LEAK_KEYS = ("threat_type", "why_this_label", "label_evidence", "lineage_group", "original_label")


def finite(o):
    if isinstance(o, float):
        return math.isfinite(o)
    if isinstance(o, dict):
        return all(finite(v) for v in o.values())
    if isinstance(o, list):
        return all(finite(v) for v in o)
    return True


def main(prefix, min_lineage=20):
    data = ROOT / "datasets_ready"
    tax = yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    rec_s = json.load(open(ROOT / "schemas/record.schema.json"))
    tel_s = json.load(open(ROOT / "schemas/telemetry.schema.json"))
    manifest = json.load(open(data / "manifest.json"))
    tok = P.load_tokenizer()
    ctrl = P.control_strings(tok)
    sysp = P.system_prompt()
    fails, ids, lens = [], set(), []
    per_split = defaultdict(Counter)
    groups = defaultdict(set)
    g_cls = defaultdict(set)
    families = defaultdict(set)
    hashes = defaultdict(set)

    for fname, info in manifest["files"].items():
        if not fname.startswith(prefix):
            continue
        p = data / fname
        if hashlib.sha256(p.read_bytes()).hexdigest() != info["sha256"]:
            fails.append(f"checksum mismatch {fname}")
        n = 0
        for ln, line in enumerate(open(p), 1):
            n += 1
            where = f"{fname}:{ln}"
            try:
                r = json.loads(line)
                jsonschema.validate(r, rec_s)
                m, msgs = r["metadata"], r["messages"]
                cls = m["threat_type"]
                if r["id"] in ids:
                    fails.append(f"dup id {where}")
                ids.add(r["id"])
                if cls not in tax:
                    fails.append(f"unknown class {cls} {where}")
                    continue
                if json.loads(msgs[2]["content"]) != {"threat_type": cls}:
                    fails.append(f"assistant target != metadata label {where}")
                if msgs[0]["content"] != sysp:
                    fails.append(f"system prompt hash drift {where}")
                tel = P.deserialize_telemetry(msgs[1]["content"])
                jsonschema.validate(tel, tel_s)
                if tel["mode"] not in tax[cls]["input_modes"]:
                    fails.append(f"mode {tel['mode']} not permitted for {cls} {where}")
                if not finite(tel):
                    fails.append(f"nonfinite value {where}")
                if any(k in msgs[1]["content"] for k in LEAK_KEYS) or cls in msgs[1]["content"]:
                    fails.append(f"label/annotation leak into model input {where}")
                if not m.get("license_ref") or not m.get("provenance"):
                    fails.append(f"missing provenance/license {where}")
                lens.append(P.check_record(tok, r, ctrl))  # mask, turns, escapes, budget
                per_split[m["split"]][cls] += 1
                groups[m["lineage_group"]].add(m["split"])
                g_cls[m["lineage_group"]].add(cls)  # a group may span classes (parent/derived child)
                families[cls].add(m["template_family"])
                hashes[hashlib.sha256(msgs[1]["content"].encode()).hexdigest()].add(m["split"])
            except (AssertionError, jsonschema.ValidationError, ValueError, KeyError) as e:
                fails.append(f"{where}: {type(e).__name__}: {str(e)[:120]}")
        if n != info["records"]:
            fails.append(f"count mismatch {fname}: {n} != {info['records']}")

    top = defaultdict(Counter)
    for split, c in per_split.items():
        top[split.split("_")[0]].update(c)
    for split, c in top.items():
        missing = sorted(set(tax) - set(c))
        if missing:
            fails.append(f"{split} missing classes: {missing}")
    # independence gate: distinct lineage groups per class (spec §4: lineage counts, not row counts)
    cls_groups = defaultdict(set)
    for g in groups:
        for c in g_cls[g]:
            cls_groups[c].add(g)
    lineage = {c: len(cls_groups[c]) for c in tax}
    thin = {c: n for c, n in lineage.items() if n < min_lineage}
    if thin:
        fails.append(f"classes below {min_lineage} independent lineage groups: {thin}")
    lo = sum(len(s) > 1 for s in groups.values())
    xo = sum(len(s) > 1 for s in hashes.values())
    if lo:
        fails.append(f"{lo} lineage groups straddle splits")
    if xo:
        fails.append(f"{xo} identical inputs across splits")

    lens.sort()
    rep = {"prefix": prefix, "passed": not fails, "n_fail": len(fails), "failures": fails[:50], "failure_kinds": dict(Counter(f.split(":")[-2].strip() if f.count(":") >= 2 else f.split(" ")[0] for f in fails).most_common(10)),
           "records": len(ids), "lineage_groups_per_class": lineage,
           "template_families_per_class": {c: sorted(families[c]) for c in tax}, "per_split": {k: dict(v) for k, v in per_split.items()},
           "token_len": {"min": lens[0], "p50": lens[len(lens) // 2], "p95": lens[int(len(lens) * .95)],
                         "max": lens[-1], "budget": P.MAX_TOKENS} if lens else None,
           "tokenizer_fingerprint": P.fingerprint(tok),
           "overlap_statement": "zero detected cross-split overlap under the documented checks "
                                "(exact input hash + lineage group)" if not (lo or xo) else "OVERLAP DETECTED"}
    (ROOT / "reports/pre_colab_gate.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("passed", "n_fail", "records", "token_len")}, indent=1))
    for f in fails[:15]:
        print("FAIL", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pilot_")
    ap.add_argument("--min-lineage", type=int, default=20)
    a = ap.parse_args()
    sys.exit(main(a.prefix, a.min_lineage))

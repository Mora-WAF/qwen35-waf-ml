"""File-based job ledger for Haiku generation subagents (spec §6, adapted per user 2026-10-15:
no paid API — generation runs as in-session Claude Code Haiku subagents, not an API worker pool).

- write_jobs(): emit one job spec per gap class into data/jobs/<class>.job.json
- A Haiku subagent reads a job, writes data/jobs/out/<class>.jsonl (telemetry objects + scenario)
- ingest(): validate each emitted object against telemetry.schema.json and wrap into records
  with label_evidence.method == scenario_oracle, generator == haiku, needs_review == true.

Idempotent: a job is DONE when its out file validates to >= count records; re-ingest is safe.
Agreement of the generator with itself is NOT ground truth; §6 label review stays separate.
"""
import json
import sys
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "data" / "jobs"
OUT = JOBS / "out"

# L7 classes with no line-seed source -> scenario fixtures from a Haiku worker.
GAP_L7 = ["ssrf_cloud_metadata", "graphql_abuse", "jwt_auth_bypass", "bola_idor_enumeration",
          "websocket_grpc_abuse", "llm_prompt_injection", "prototype_pollution",
          "web_cache_poisoning", "deserialization", "csv_formula_injection"]
GAP_BENIGN = ["benign_hard_negative_dev", "benign_hard_negative_search"]
GAP = GAP_L7 + GAP_BENIGN
EXTRA_INGEST = ["benign_standard_traffic"]  # context-matched benign counterparts (v1 file)
GENERIC_HEADERS = {"host", "user-agent", "accept", "accept-language", "accept-encoding", "connection",
                   "dnt", "upgrade-insecure-requests"}  # + sec-fetch-* / sec-ch-ua*


def taxonomy(root=ROOT):
    return yaml.safe_load(open(root / "config/taxonomy.yaml"))["classes"]


def write_jobs(classes, count, root=ROOT):
    JOBS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tax = taxonomy(root)
    prompt = (root / "subagents/prompts/scenario_v1.md").read_text()
    written = []
    for cls in classes:
        spec = tax[cls]
        job = {"class": cls, "count": count, "domain": spec["domain"],
               "input_modes": spec["input_modes"], "scope": spec["scope"],
               "exclusions": spec["exclusions"], "required_evidence": spec["required_evidence"],
               "benign_counterexamples": spec["benign_counterexamples"],
               "is_benign": spec["is_benign"],
               "out_path": str((OUT / f"{cls}.jsonl").relative_to(root)),
               "telemetry_schema": "schemas/telemetry.schema.json",
               "instructions": prompt}
        p = JOBS / f"{cls}.job.json"
        p.write_text(json.dumps(job, indent=1))
        written.append(str(p))
    return written


def _neutralize(tel, key):
    """Apply the shared hostname-shortcut rewrite to every string in a telemetry object."""
    from data_pipeline.generate import neutralize_hosts, rng_for
    r, changed = rng_for(key), False

    def walk(o):
        nonlocal changed
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str):
            o, c = neutralize_hosts(o, r)
            changed |= c
        return o
    return walk(tel), changed


def near_dup_group(cls, tel):
    """Lineage = digit/hex-masked telemetry shape. Script-templated worker output that differs
    only in IDs/numbers collapses to one group, so near-duplicates never straddle splits (§8)."""
    import hashlib
    import re
    shape = re.sub(r"[0-9a-fA-F]{6,}|\d+", "#", json.dumps(tel, sort_keys=True))
    return f"haiku:{cls}:{hashlib.sha256(shape.encode()).hexdigest()[:12]}"


def normalize_headers(tel, key):
    """Replace generator-specific generic headers (UA, Accept*, Host...) with the shared pool profile
    used by deterministic records, so header style no longer identifies the generator (spec §8).
    Semantic headers (Authorization, Origin, Content-Type, X-Forwarded-*, Cache-*, Upgrade...) kept."""
    from data_pipeline.generate import HOSTS, _headers, rng_for
    h = tel.get("http")
    if not h:
        return tel
    r = rng_for("hdr:" + key)
    keep = [kv for kv in h["headers"] if kv[0].lower() not in GENERIC_HEADERS
            and not kv[0].lower().startswith(("sec-fetch-", "sec-ch-ua"))]
    h["headers"] = _headers(r, r.choice(HOSTS)) + keep
    return tel


def ingest(root=ROOT):
    from data_pipeline.generate import assemble
    from training.prepare import deserialize_telemetry  # noqa: F401 (round-trip sanity imported in tests)
    tel_schema = json.load(open(root / "schemas/telemetry.schema.json"))
    tax = taxonomy(root)
    all_recs, report = [], {}
    for cls in GAP + EXTRA_INGEST:
        fps = sorted(OUT.glob(f"{cls}.jsonl")) + sorted(OUT.glob(f"{cls}.v*.jsonl"))  # v1 + later rounds
        if not fps:
            report[cls] = {"status": "missing", "accepted": 0}
            continue
        accepted, rejected = [], 0
        from subagents.review import status
        rv = status()
        lines = [(f"{fp.name}:{j}", ln) for fp in fps for j, ln in enumerate(open(fp))]
        quarantined = 0
        for i, (ref, line) in enumerate(lines):
            if rv.get(ref) == "quarantined":  # blind reviewer disagreed / evidence insufficient
                quarantined += 1
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tel, hn = _neutralize(obj["telemetry"], f"{cls}:{i}")
                tel = normalize_headers(tel, f"{cls}:{i}")
                jsonschema.validate(tel, tel_schema)
                assert tel["mode"] in tax[cls]["input_modes"] or tel["mode"] == "HYBRID_CTX"
                rec_id = f"haiku:{cls}:{i:06d}"
                rec = assemble(rec_id, cls, tel, "synthetic",
                               {"method": "scenario_oracle", "oracle": "haiku_scenario",
                                "scenario": obj.get("scenario", {}), "needs_review": rv.get(ref) != "agreed",
                                "blind_review": rv.get(ref, "not_reviewed"),
                                "transforms": (["host_neutralize"] if hn else []) + ["header_profile_normalize"]},
                               {"lineage_group": near_dup_group(cls, tel), "generator": "haiku-5.5",
                                "template_family": f"{cls}_scenario"})
                accepted.append(rec)
            except (json.JSONDecodeError, KeyError, jsonschema.ValidationError, AssertionError):
                rejected += 1
        all_recs += accepted
        report[cls] = {"status": "ok" if accepted else "empty", "accepted": len(accepted),
                       "rejected": rejected, "quarantined_by_review": quarantined}
    return all_recs, report


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    if sys.argv[1:2] == ["jobs"]:
        for p in write_jobs(GAP, n):
            print("job:", p)
    elif sys.argv[1:2] == ["ingest"]:
        recs, rep = ingest()
        print(json.dumps(rep, indent=1))
        print("total accepted:", len(recs))

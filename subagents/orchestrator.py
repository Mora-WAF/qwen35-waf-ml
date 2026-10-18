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
import re
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
          "web_cache_poisoning", "deserialization", "csv_formula_injection",
          "bopla_mass_assignment", "bfla_admin_function_probing", "race_condition_toctou_burst",
          "malicious_file_upload_webshell", "mcp_agentic_tool_abuse"]
GAP_BENIGN = ["benign_hard_negative_dev", "benign_hard_negative_search"]
GAP = GAP_L7 + GAP_BENIGN
EXTRA_INGEST = ["benign_standard_traffic",  # context-matched benign counterparts (v1 file)
                "xss_reflected_stored", "xxe",  # confuser-set records next to the seeded ones (prompt confusers_v1.md)
                "benign_high_rate_legit",  # hard twins of rate-lookalike classes (prompt hard_twins_v1.md)
                # Codex scenario families for seeded classes (round vcx): named families make test_unseen possible
                "crlf_header_injection", "http_request_smuggling", "ldap_xpath_injection", "nosqli", "open_redirect_cors",
                "path_traversal_lfi", "rce_cmd_injection", "ssti", "xss_dom_polyglot",
                "sqli_classic"]  # 18 Oct: nuclei-style and SOAP rounds (vcx74/vgm74, SOAP round) were not ingested
# injection syntax in a generic header: the header is the payload carrier, so it survives normalize_headers
HEADER_PAYLOAD = re.compile(r"\$\{|\(\)\s*\{|[<>`]|\$\(|\.\./|%[0-9a-fA-F]{2}|\|\s*[a-z]")
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


# v2 jobs: canonical context key names per class (from the v1 outputs, verdict-like keys removed)
CONTEXT_SCHEMA = {
    "ssrf_cloud_metadata": ["context.fetch_config", "context.resolution"],
    "graphql_abuse": ["context.parsed_operation", "context.schema_config", "history.same_operation_count_60s", "history.operation_hash"],
    "jwt_auth_bypass": ["context.verifier_config", "context.token_material", "context.verifier_result"],
    "bola_idor_enumeration": ["context.authz_context", "context.authz_policy", "history.prior_requests", "history.window_s"],
    "websocket_grpc_abuse": ["context.stream", "context.origin_policy", "context.peer", "history.subscribe_frames", "history.open_streams_from_peer"],
    "llm_prompt_injection": ["context.trust_boundary", "context.task", "context.untrusted_text_location"],
    "prototype_pollution": ["context.runtime", "context.merge_sink", "context.raw_body_key_path"],
    "web_cache_poisoning": ["context.cache_config", "context.unkeyed_input", "context.cache_status_samples"],
    "deserialization": ["context.parser_context", "context.fetch_behavior"],
    "csv_formula_injection": ["context.sink_context"],
    "bopla_mass_assignment": ["context.object_schema", "context.caller_role", "context.write_result"],
    "bfla_admin_function_probing": ["context.caller_role", "context.route_required_role", "history.prior_requests", "history.window_s"],
    "race_condition_toctou_burst": ["context.route_semantics", "history.burst_window_ms", "history.same_request_count",
                                    "history.distinct_idempotency_keys", "history.response_outcomes"],
    "malicious_file_upload_webshell": ["context.upload_handler", "context.storage_path_policy", "context.content_inspection"],
    "mcp_agentic_tool_abuse": ["context.tool_manifest", "context.trust_boundary", "context.agent_task", "context.tool_call"],
    "benign_hard_negative_dev": [], "benign_hard_negative_search": [],
}


def existing_families(cls):
    """Family names already generated for a class: later rounds must add new families, not more of these."""
    fams = set()
    for fp in sorted(OUT.glob(f"{cls}.v*.jsonl")):
        for line in open(fp):
            if line.strip():
                f = (json.loads(line).get("scenario") or {}).get("family")
                if f:
                    fams.add(f)
    return sorted(fams)


def write_jobs_v2(classes, count, families, version, root=ROOT, focus=None, focus_domains=None):
    """v2 job specs: >= `families` scenario families per class and, for attack classes, a context-matched
    benign twin file (benign_standard_traffic.<version>_<class>.jsonl). Raw outputs are new files only.
    `focus`: one evidence pattern every record must follow (targeted top-up for a measured model error)."""
    tax = taxonomy(root)
    prompt = (root / "subagents/prompts/scenario_v2.md").read_text()
    written = []
    for cls in classes:
        spec = tax[cls]
        job = {"class": cls, "count": count, "families": families, "domain": spec["domain"],
               "input_modes": spec["input_modes"], "scope": spec["scope"], "exclusions": spec["exclusions"],
               "required_evidence": spec["required_evidence"], "benign_counterexamples": spec["benign_counterexamples"],
               "is_benign": spec["is_benign"], "context_schema": CONTEXT_SCHEMA.get(cls, []),
               "avoid_families": existing_families(cls),
               "out_path": str((OUT / f"{cls}.{version}.jsonl").relative_to(root)),
               "telemetry_schema": "schemas/telemetry.schema.json", "instructions": prompt}
        job |= {k: v for k, v in (("focus", focus), ("focus_domains", focus_domains)) if v}
        if not spec["is_benign"]:
            job["twin_out_path"] = str((OUT / f"benign_standard_traffic.{version}_{cls}.jsonl").relative_to(root))
        for k in ("out_path", "twin_out_path"):
            assert k not in job or not (root / job[k]).exists(), f"refusing to overwrite raw output {job[k]}"
        p = JOBS / f"{cls}.{version}.job.json"
        p.write_text(json.dumps(job, indent=1))
        written.append(str(p))
    return written


DOMAINS = ["e-commerce storefront", "retail banking", "payments/fintech API", "healthcare patient portal",
           "insurance claims", "SaaS admin console", "IoT device management API", "media/CMS publishing",
           "government services", "university LMS", "logistics/shipping tracker", "online gaming platform",
           "HR/internal tools", "CI/CD and developer tooling", "telecom self-care", "travel booking",
           "real-estate listings", "food delivery", "crypto exchange", "B2B partner integration",
           "customer support desk", "social network", "ad-tech/analytics", "energy/utility billing"]


def write_jobs_scale(target_frac=0.25, per_worker=96, families=12, twin_count=10, tag="s1", slack=1.08):
    """Scale round: enough workers per Haiku class to reach target_frac x quota accepted records.
    Each worker gets a distinct application-domain focus so parallel workers do not converge on the same
    families; twins are capped (twin_count per worker) because benign_standard_traffic has its own quota."""
    import math
    from collections import Counter
    tax = taxonomy()
    have = Counter(r["metadata"]["threat_type"] for r in ingest()[0])
    prompt = (ROOT / "subagents/prompts/scenario_v2.md").read_text()
    jobs = []
    for cls in GAP:
        need = max(0, math.ceil(tax[cls]["quota"] * target_frac) - have[cls])
        n_workers = math.ceil(need * slack / per_worker)
        avoid = existing_families(cls)
        for k in range(n_workers):
            ver = f"v{tag}w{k:02d}"
            spec = tax[cls]
            job = {"class": cls, "count": per_worker, "families": families, "domain": spec["domain"],
                   "input_modes": spec["input_modes"], "scope": spec["scope"], "exclusions": spec["exclusions"],
                   "required_evidence": spec["required_evidence"], "benign_counterexamples": spec["benign_counterexamples"],
                   "is_benign": spec["is_benign"], "context_schema": CONTEXT_SCHEMA.get(cls, []), "avoid_families": avoid,
                   "focus_domains": [DOMAINS[(2 * k + i) % len(DOMAINS)] for i in range(2)],
                   "out_path": str((OUT / f"{cls}.{ver}.jsonl").relative_to(ROOT)),
                   "telemetry_schema": "schemas/telemetry.schema.json", "instructions": prompt}
            if not spec["is_benign"]:
                job["twin_count"] = twin_count
                job["twin_out_path"] = str((OUT / f"benign_standard_traffic.{ver}_{cls}.jsonl").relative_to(ROOT))
            for key in ("out_path", "twin_out_path"):
                assert key not in job or not (ROOT / job[key]).exists(), f"refusing to overwrite {job[key]}"
            p = JOBS / f"{cls}.{ver}.job.json"
            p.write_text(json.dumps(job, indent=1))
            jobs.append(str(p.relative_to(ROOT)))
    return jobs


VERDICT_KEY = re.compile(r'"(malicious|attack|suspicious|override_match|injection_detected|is_attack|threat_type)"', re.I)


def verify(paths):
    """Check worker output files before review: line counts vs job, schema, family spread, verdict keys,
    duplicates against every other output, and near-duplicate share (script-templated output collapses
    into few near_dup_group shapes). Returns {path: problems}; empty list = ok."""
    tel_schema = json.load(open(ROOT / "schemas/telemetry.schema.json"))
    seen = {}
    for fp in sorted(OUT.glob("*.jsonl")):
        if str(fp.relative_to(ROOT)) in paths:
            continue
        for line in open(fp):
            if line.strip():
                seen[json.dumps(json.loads(line)["telemetry"], sort_keys=True)] = fp.name
    out = {}
    for path in paths:
        fp, probs = ROOT / path, []
        if not fp.exists():
            out[path] = ["missing"]
            continue
        rows = [json.loads(x) for x in open(fp) if x.strip()]
        check = jsonschema.validators.validator_for(tel_schema)(tel_schema).validate
        fams, shapes = set(), set()
        cls = fp.name.split(".")[0]
        for o in rows:
            try:
                check(o["telemetry"])
            except jsonschema.ValidationError:
                probs.append("schema")
            t = json.dumps(o["telemetry"], sort_keys=True)
            if VERDICT_KEY.search(t):
                probs.append("verdict_key")
            if t in seen:
                probs.append(f"dup:{seen[t]}")
            seen[t] = fp.name
            fams.add((o.get("scenario") or {}).get("family"))
            shapes.add(near_dup_group(cls, o["telemetry"]))
        out[path] = sorted(set(probs)) + ([f"near_dup_share={1 - len(shapes) / len(rows):.2f}"]
                                          if rows and len(shapes) / len(rows) < 0.8 else [])
        out[path].append(f"n={len(rows)} families={len(fams - {None})}")
    return out


def write_twin_jobs(target_ratio=0.35, per_worker=96, slack=1.15, tag="tw"):
    """Twin-only jobs: benign twins for already-generated, review-agreed attack lines that have none,
    until each Haiku attack class has ~target_ratio twins per accepted attack record."""
    import math
    from subagents.review import status
    tax, rv = taxonomy(), status()
    prompt = (ROOT / "subagents/prompts/twins_v1.md").read_text()
    jobs, k = [], 0
    for cls in GAP_L7:
        twinned, atk, tw = set(), [], 0
        for fp in OUT.glob(f"benign_standard_traffic.*_{cls}.jsonl"):
            ver = fp.name.split(".v", 1)[1].split("_", 1)[0]
            for i, line in enumerate(open(fp)):
                if not line.strip():
                    continue
                o = json.loads(line)
                t = (o.get("scenario") or {}).get("twin_of", o.get("twin_of"))
                twinned.add(t if isinstance(t, str) else f"{cls}.v{ver}.jsonl:{t}")
                tw += rv.get(f"{fp.name}:{i}") != "quarantined"
        for fp in sorted(OUT.glob(f"{cls}.v*.jsonl")):
            for i, line in enumerate(open(fp)):
                ref = f"{fp.name}:{i}"
                if line.strip() and rv.get(ref) == "agreed":
                    atk.append(ref)
        need = math.ceil(max(0, target_ratio * len(atk) - tw) * slack)
        todo = [r for r in atk if r not in twinned][:need]
        by_file = {}
        for r in todo:
            f, i = r.rsplit(":", 1)
            by_file.setdefault(f, []).append(int(i))
        chunks = [(f, ls[j:j + per_worker]) for f, ls in sorted(by_file.items()) for j in range(0, len(ls), per_worker)]
        for f, ls in chunks:
            job = {"class": cls, "source_path": str((OUT / f).relative_to(ROOT)), "source_lines": ls,
                   "context_schema": CONTEXT_SCHEMA[cls], "scope": tax[cls]["scope"],
                   "benign_counterexamples": tax[cls]["benign_counterexamples"],
                   "out_path": str((OUT / f"benign_standard_traffic.v{tag}{k:03d}_{cls}.jsonl").relative_to(ROOT)),
                   "telemetry_schema": "schemas/telemetry.schema.json", "instructions": prompt}
            assert not (ROOT / job["out_path"]).exists(), job["out_path"]
            p = JOBS / f"twins.{tag}{k:03d}_{cls}.job.json"
            p.write_text(json.dumps(job, indent=1))
            jobs.append(str(p.relative_to(ROOT)))
            k += 1
    return jobs


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
    keep = [kv for kv in h["headers"] if (kv[0].lower() not in GENERIC_HEADERS
            and not kv[0].lower().startswith(("sec-fetch-", "sec-ch-ua")))
            or (kv[0].lower() != "host" and HEADER_PAYLOAD.search(str(kv[1])))]  # the evidence itself (Log4j UA, Shellshock)
    keep = [kv for kv in keep if not kv[0].lower().startswith("x-fixture")]  # generator marker headers (18 Oct, Sol)
    kept = {k.lower() for k, _ in keep}
    h["headers"] = [kv for kv in _headers(r, r.choice(HOSTS)) if kv[0].lower() not in kept] + keep
    return tel


# context keys that state a verdict or annotation (or exist only in one generator's records) rather than an
# observation a sensor would log: dropped at ingest so they cannot become label shortcuts
STRIP_CONTEXT_KEYS = {"override_match", "payload_feature", "endpoint", "client", "verdict", "label", "why",
                      "is_malicious", "suspicious", "detected", "attack_type"}
# v1 synonyms of the v2 canonical context names (CONTEXT_SCHEMA); renamed when the canonical key is absent
LEGACY_RENAME = {"sink": "merge_sink", "key_path": "raw_body_key_path", "raw_key_structure": "raw_body_key_path"}
# placeholder hosts workers put in attack payloads only ("attacker"/"evil" became attack-only tokens, 18 Oct)
PLACEHOLDER_HOST = re.compile(r"(?i)(attacker|evil)\.example\b")  # no leading \b: %2Fevil.example
NEUTRAL_HOSTS = ["cdn-metrics.example.net", "img-cache.example.org", "sync.example.com", "telemetry.example.io",
                 "edge-07.example.net", "assets.example.org"]
OBSERVED_UNTIL = {"flow_window": "fixed_window", "flow_complete": "full_flow", "session_history": "session_window"}


def canonicalize(tel, key):
    """Remove generator style from fields that carry no evidence: Haiku wrote ISO timestamps into
    observed_until and 'HTTP/2.0' where deterministic records write stage tokens and 'HTTP/2'
    (both were near-perfect generator -> class shortcuts). Then add the shared neutral context."""
    from data_pipeline.generate import neutral_context, rng_for
    tel["observed_until"] = OBSERVED_UNTIL.get(tel["decision_stage"], "request_received")
    for part in ("context", "history"):
        if isinstance(tel.get(part), dict):
            kept = {}
            for k, v in tel[part].items():
                if k.lower() in STRIP_CONTEXT_KEYS:
                    continue
                k = LEGACY_RENAME.get(k, k) if LEGACY_RENAME.get(k) not in tel[part] else k
                kept.setdefault(k, v)
            tel[part] = kept or None
            if tel[part] is None:
                del tel[part]
    h = tel.get("http")
    if h:
        h["protocol"] = {"HTTP/2.0": "HTTP/2", "HTTP/3.0": "HTTP/3", "http/1.1": "HTTP/1.1"}.get(h["protocol"], h["protocol"])
    s = json.dumps(tel)
    if PLACEHOLDER_HOST.search(s):
        host = rng_for("host:" + key).choice(NEUTRAL_HOSTS)
        tel = json.loads(PLACEHOLDER_HOST.sub(host, s))
    return neutral_context(rng_for("ctx:" + key), tel)


def generator_of(fname):
    """Raw output file -> generator name, from the round tag in the file name (vsn = Sonnet 5.5 workers,
    vgm = Gemini 3.8 Flash medium via subagents.gen_http)."""
    if re.search(r"\.v(ht|cf|cx)\d", fname):
        return "codex-gpt-6.1-sol"
    if re.search(r"\.vgm\d", fname):
        return "gemini-3.8-flash"
    return "sonnet-5.5" if re.search(r"\.vsn\d", fname) else "haiku-5.5"


def ingest(root=ROOT):
    from data_pipeline.generate import assemble
    from training.prepare import deserialize_telemetry  # noqa: F401 (round-trip sanity imported in tests)
    tel_schema = json.load(open(root / "schemas/telemetry.schema.json"))
    check = jsonschema.validators.validator_for(tel_schema)(tel_schema).validate  # compiled once, not per record
    tax = taxonomy(root)
    all_recs, report = [], {}
    _lines = {}

    def file_lines(p):  # each twin source file read once, not once per twin line
        if p not in _lines:
            _lines[p] = [x for x in open(p) if x.strip()] if p.exists() else []
        return _lines[p]
    for cls in GAP + EXTRA_INGEST:
        fps = sorted(OUT.glob(f"{cls}.jsonl")) + sorted(OUT.glob(f"{cls}.v*.jsonl"))  # v1 + later rounds
        if not fps:
            report[cls] = {"status": "missing", "accepted": 0}
            continue
        accepted, rejected = [], 0
        from subagents.review import relabels, status
        rv, rel = status(), relabels()
        qp = ROOT / "data/review/quarantine.json"
        qrev = {x["ref"]: (x["reviewer"], x["sufficient"]) for x in json.load(open(qp))} if qp.exists() else {}
        lines = [(f"{fp.name}:{j}", ln) for fp in fps for j, ln in enumerate(open(fp))]
        quarantined = 0
        for i, (ref, line) in enumerate(lines):
            # a benign->benign relabel outranks the Haiku quarantine when Haiku itself chose that benign class
            # (with sufficient evidence): the label moves between benign classes, it is not dropped
            relabel_agreed = rel.get(ref) not in (None, "__quarantine__") and qrev.get(ref) == (rel[ref], True)
            if (rv.get(ref) == "quarantined" and not relabel_agreed) or rel.get(ref) == "__quarantine__":
                quarantined += 1
                continue
            out_cls = rel.get(ref, cls)  # external-review relabel between benign classes (subagents.review.apply_hn)
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tel, hn = _neutralize(obj["telemetry"], f"{cls}:{i}")
                tel = canonicalize(normalize_headers(tel, f"{cls}:{i}"), f"{cls}:{i}")
                check(tel)
                assert tel["mode"] in tax[cls]["input_modes"] or tel["mode"] == "HYBRID_CTX"
                rec_id = f"haiku:{cls}:{i:06d}"
                group = near_dup_group(cls, obj["telemetry"])
                twin_of = (obj.get("scenario") or {}).get("twin_of", obj.get("twin_of"))  # workers put it in either place
                fname = ref.split(":")[0]
                if isinstance(twin_of, str) and ":" in twin_of:  # twin-only jobs name the exact source line
                    src_name, src_line = twin_of.rsplit(":", 1)
                    src_lines = file_lines(OUT / src_name)
                    if src_line.isdigit() and int(src_line) < len(src_lines):
                        group = near_dup_group(src_name.split(".")[0], json.loads(src_lines[int(src_line)])["telemetry"])
                elif twin_of is not None and ".v" in fname and "_" in fname.split(".v", 1)[1]:
                    # context-matched benign twin: same lineage group as its attack record (never split apart)
                    atk_cls = fname.split(".v", 1)[1].split("_", 1)[1].removesuffix(".jsonl")
                    atk_file = OUT / f"{atk_cls}.v{fname.split('.v', 1)[1].split('_', 1)[0]}.jsonl"
                    atk_lines = file_lines(atk_file)
                    if 0 <= int(twin_of) < len(atk_lines):
                        group = near_dup_group(atk_cls, json.loads(atk_lines[int(twin_of)])["telemetry"])
                rec = assemble(rec_id, out_cls, tel, "synthetic",
                               {"method": "scenario_oracle", "oracle": "haiku_scenario",
                                "scenario": obj.get("scenario", {}), "needs_review": rv.get(ref) != "agreed",
                                "blind_review": rv.get(ref, "not_reviewed"),
                                "transforms": (["host_neutralize"] if hn else []) + ["header_profile_normalize"]},
                               {"lineage_group": group, "generator": generator_of(fname),
                                "template_family": f"{cls}:{(obj.get('scenario') or {}).get('family', 'v1_unlabelled')}"})
                accepted.append(rec)
            except (json.JSONDecodeError, KeyError, jsonschema.ValidationError, AssertionError):
                rejected += 1
        all_recs += accepted
        report[cls] = {"status": "ok" if accepted else "empty", "accepted": len(accepted),
                       "rejected": rejected, "quarantined_by_review": quarantined}
    return all_recs, report


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 24
    if sys.argv[1:2] == ["jobs"]:
        for p in write_jobs(GAP, n):
            print("job:", p)
    elif sys.argv[1:2] == ["jobs_v2"]:  # jobs_v2 COUNT FAMILIES VERSION
        for p in write_jobs_v2(GAP, n, int(sys.argv[3]), sys.argv[4]):
            print("job:", p)
    elif sys.argv[1:2] == ["jobs_focus"]:  # jobs_focus CLASS COUNT FAMILIES VERSION "FOCUS" ["DOMAIN;DOMAIN"]
        for p in write_jobs_v2([sys.argv[2]], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], focus=sys.argv[6],
                               focus_domains=sys.argv[7].split(";") if len(sys.argv) > 7 else None):
            print("job:", p)
    elif sys.argv[1:2] == ["jobs_scale"]:  # jobs_scale TAG -> one job per worker, listed in data/jobs/scale_TAG.txt
        jobs = write_jobs_scale(tag=sys.argv[2])
        (JOBS / f"scale_{sys.argv[2]}.txt").write_text("\n".join(jobs) + "\n")
        print(len(jobs), "jobs")
    elif sys.argv[1:2] == ["verify"]:  # verify FILE... (paths relative to the repo root)
        for k, v in verify(sys.argv[2:]).items():
            print(k, v)
    elif sys.argv[1:2] == ["ingest"]:
        recs, rep = ingest()
        print(json.dumps(rep, indent=1))
        print("total accepted:", len(recs))

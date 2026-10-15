"""OWASP CRS regression fixtures -> annotated HTTP telemetry pool (spec §3).

Positive stages only (`expect_ids`): the rule author expects that rule to fire. A rule match is an
annotation signal, not universal truth, so fixtures still pass the seed quality filter and blind review.
Families 930/932/941/942 map to classes; 944 is skipped (keyword-stuffed unit tests, not requests).
Lineage = rule id (all tests of one rule are variations of one pattern).

usage: python -m data_pipeline.crs   -> data/crs/pool.jsonl + reports/crs_ingest.json
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, unquote_plus, urlsplit

import yaml

from data_pipeline import generate as G
from data_pipeline.normalize import quality_reject

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT.parent / "raw_cache"
OUT = ROOT / "data" / "crs"
FAMILY = {"930": "path_traversal_lfi", "932": "rce_cmd_injection",
          "941": "xss_reflected_stored", "942": "sqli_classic"}


def telemetry(inp, rule_id, test_id):
    uri = str(inp.get("uri", "/"))
    data = inp.get("data")
    if isinstance(data, list):
        data = "\n".join(map(str, data))
    headers = [[str(k), str(v)] for k, v in (inp.get("headers") or {}).items()]
    q = urlsplit(uri).query
    http = {"method": str(inp.get("method", "GET")), "target": uri,
            "protocol": str(inp.get("version", "HTTP/1.1")), "headers": headers,
            "query": [list(kv) for kv in parse_qsl(q, keep_blank_values=True)] if q else None,
            "cookies": None, "body": None if data is None else str(data),
            "body_encoding": "none" if data is None else "utf8"}
    tel = {"schema_version": 1, "mode": "L7_HTTP",
           "decision_stage": "request_headers_body" if data is not None else "request_headers",
           "observed_until": "request_received", "truncated": False, "http": http}
    # same shortcut controls as generated records: neutral hosts / entity names, shared header profile
    from subagents.orchestrator import _neutralize, canonicalize, normalize_headers
    tel, _ = _neutralize(tel, f"crs:{rule_id}:{test_id}")
    return canonicalize(normalize_headers(tel, f"crs:{rule_id}:{test_id}"), f"crs:{rule_id}:{test_id}")


def ingest():
    src = {s["id"]: s for s in yaml.safe_load(open(ROOT / "config/sources.yaml"))["sources"]}["coreruleset"]
    base = RAW / src["id"] / src["revision"]
    pool, rep = [], defaultdict(Counter)
    for f in sorted(base.rglob("REQUEST-9*/*.yaml")):
        fam = f.parent.name.split("-")[1]
        if fam not in FAMILY:
            continue
        cls = FAMILY[fam]
        d = yaml.safe_load(open(f))
        for t in d.get("tests", []):
            for k, st in enumerate(t.get("stages", [])):
                st = st.get("stage", st)
                inp, out = st.get("input") or {}, st.get("output") or {}
                log = out.get("log") if isinstance(out.get("log"), dict) else {}
                if not log.get("expect_ids") or inp.get("encoded_request"):
                    rep[cls]["not_positive_or_encoded"] += 1
                    continue
                # the attack-bearing text: decoded query + body
                surface = unquote_plus(urlsplit(str(inp.get("uri", ""))).query) + " " + str(inp.get("data") or "")
                why = quality_reject(surface.strip(), "lines")
                if why:
                    rep[cls][why] += 1
                    continue
                tel = telemetry(inp, d["rule_id"], t["test_id"])
                pool.append({"cls": cls, "telemetry": tel, "source": src["id"], "source_revision": src["revision"],
                             "license_ref": src["license_spdx"], "original_id": f"{d['rule_id']}-{t['test_id']}-{k}",
                             "original_label": f"crs_rule_{d['rule_id']}", "lineage_group": f"crs:{d['rule_id']}"})
                rep[cls]["accepted"] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pool.jsonl", "w") as fh:
        for x in pool:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    groups = defaultdict(set)
    for x in pool:
        groups[x["cls"]].add(x["lineage_group"])
    report = {c: dict(v) | {"rules": len(groups[c])} for c, v in rep.items()}
    (ROOT / "reports/crs_ingest.json").write_text(json.dumps(report, indent=1))
    return report


def load_pool():
    p = OUT / "pool.jsonl"
    by = defaultdict(list)
    if p.exists():
        for line in open(p):
            x = json.loads(line)
            by[x["cls"]].append(x)
    return by


if __name__ == "__main__":
    print(json.dumps(ingest(), indent=1))

"""Independent, non-LLM label cross-check: do OWASP CRS rule-family regexes fire on a record?

Generation and review are both Claude Haiku, so their agreement is not independent. CRS is a separate
human-written detector. This is an approximate evaluation: only `@rx` operators, compiled with Python
`re` (patterns that do not compile are skipped and counted), applied to lower-cased, URL-decoded request
fields. No chains, no paranoia-level gating, no anomaly scoring. It yields a cross-check signal and a list
of disagreements to inspect, never a label.

usage: python -m eval.crs_oracle [--prefix pilot_]   -> reports/crs_oracle.json
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote_plus

import yaml

ROOT = Path(__file__).resolve().parent.parent
FAMILY = {"sqli_classic": ["942"], "sqli_obfuscated": ["942"], "nosqli": ["942", "934"],
          "xss_reflected_stored": ["941"], "xss_dom_polyglot": ["941"], "rce_cmd_injection": ["932"],
          "path_traversal_lfi": ["930"], "ssti": ["934", "932", "933"], "crlf_header_injection": ["921"],
          "http_request_smuggling": ["921"], "deserialization": ["944", "933", "934"],
          "prototype_pollution": ["934"]}
RX = re.compile(r'SecRule\s+(\S+)\s+"@rx\s+((?:[^"\\]|\\.)*)"\s*\\?\s*"([^"]*)"', re.S)
MAX_PL = 2  # attacks: PL<=2 (PL1 SQLi relies on libinjection, not @rx); benign: PL1 only (default profile)


def load_rules():
    src = {s["id"]: s for s in yaml.safe_load(open(ROOT / "config/sources.yaml"))["sources"]}["coreruleset_rules"]
    base = ROOT.parent / "raw_cache" / src["id"] / src["revision"]
    rules, skipped = defaultdict(list), 0
    for f in src["files"]:
        text = (base / f["path"]).read_text()
        fam = f["path"].split("REQUEST-")[1][:3]
        for targets, pat, actions in RX.findall(text):
            rid = re.search(r"id:(\d+)", actions)
            pl = re.search(r"paranoia-level/(\d)", actions)
            if (not rid or "chain" in actions or (pl and int(pl.group(1)) > MAX_PL)
                    or "block" not in actions):  # pass/setvar-only rules are counters, not detections
                continue  # chained first conditions alone are weaker than the rule; high PL over-fires
            where = {"headers"} if "REQUEST_HEADERS" in targets and "ARGS" not in targets else {"args"}
            if "REQUEST_HEADERS" in targets and "ARGS" in targets:
                where = {"args", "headers"}
            try:
                rules[fam].append((rid.group(1), re.compile(pat.replace('\\"', '"'), re.I | re.S), where,
                                   int(pl.group(1)) if pl else 1))
            except re.error:
                skipped += 1
    return rules, skipped


def _decode(out):
    dec = []
    for s in out:
        for _ in range(2):
            s2 = unquote_plus(s)
            dec.append(s2.lower())
            if s2 == s:
                break
            s = s2
    return dec


def fields(tel):
    """args = parameter-like values (path, query, body, cookies, decoded views); headers = header values
    other than the client's own identification headers."""
    h = tel.get("http") or {}
    args = [h.get("target", "").split("?")[0], h.get("body") or ""]
    args += [v for _, v in (h.get("query") or []) + (h.get("cookies") or [])]
    args += [v["value"] for v in h.get("decoded_views", [])]
    hdrs = [v for k, v in h.get("headers", []) if k.lower() not in ("user-agent", "accept", "accept-language",
                                                                      "accept-encoding", "host")]
    return {"args": _decode(args), "headers": _decode(hdrs)}


def main(prefix):
    rules, skipped = load_rules()
    tax = yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    hits, n, silent, fires_benign = Counter(), Counter(), [], []
    benign_fam, benign_rules = Counter(), Counter()
    for fn in sorted((ROOT / "datasets_ready").glob(f"{prefix}*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            cls = r["metadata"]["threat_type"]
            tel = json.loads(r["messages"][1]["content"].split("] ", 1)[1])
            if not tel.get("http"):
                continue
            vals = fields(tel)
            fams = FAMILY.get(cls) or (["921", "930", "931", "932", "933", "934", "941", "942", "944"] if tax[cls]["is_benign"] else [])
            if not fams:
                continue
            max_pl = 1 if tax[cls]["is_benign"] else MAX_PL
            fired = [(f, rid) for f in fams for rid, rx, where, pl in rules.get(f, [])
                     if pl <= max_pl and any(rx.search(v) for w in where for v in vals[w])]
            n[cls] += 1
            if tax[cls]["is_benign"]:
                if fired:
                    hits[cls] += 1
                    benign_fam.update({f for f, _ in fired})
                    benign_rules.update({rid for _, rid in fired})
                    fires_benign.append({"id": r["id"], "class": cls, "rules": sorted({rid for _, rid in fired})[:5]})
            elif fired:
                hits[cls] += 1
            else:
                silent.append({"id": r["id"], "class": cls, "provenance": r["metadata"]["provenance"]})
    rep = {"prefix": prefix, "rules_compiled": {f: len(v) for f, v in rules.items()}, "rules_skipped": skipped,
           "attack_family_hit_rate": {c: round(hits[c] / n[c], 2) for c in sorted(n) if not tax[c]["is_benign"]},
           "benign_any_rule_fire_rate": {c: round(hits[c] / n[c], 2) for c in sorted(n) if tax[c]["is_benign"]},
           "benign_fires_by_family": dict(benign_fam), "benign_top_rules": benign_rules.most_common(10),
           "attack_records_crs_silent": silent, "benign_records_crs_fires": fires_benign,
           "caveat": "approximate @rx-only CRS evaluation; silence or firing is a prompt to inspect, not a label. "
                     "Low hit rates are expected for classes CRS does not target well (context-only evidence)."}
    (ROOT / "reports/crs_oracle.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("rules_compiled", "rules_skipped", "attack_family_hit_rate",
                                          "benign_any_rule_fire_rate", "benign_fires_by_family", "benign_top_rules")}, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pilot_")
    main(ap.parse_args().prefix)

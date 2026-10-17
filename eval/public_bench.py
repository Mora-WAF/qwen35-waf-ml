"""Public benchmarks for the deployed v10 model (evaluation only: none of these sources is training data).

  gotestwaf   Wallarm GoTestWAF test cases (MIT, commit 8ed7bc21): payload x encoder x placeholder, built the way
              GoTestWAF builds its requests; plus its false-positive text set. RawRequest/GraphQL/gRPC cases skipped.
  openappsec  open-appsec WAF Comparison Project: legitimate browsing traffic (Apache-2.0, sampled per site) and
              the malicious set (MGM WAF payload collection, mostly MIT), sampled per category.
  unsw        NF-UNSW-NB15 v1 (UQ, academic use with citation): NetFlow v1 rows. v1 has no packet-length columns,
              so pkt_len_min/max are absent (every training flow had them): a known input shift.

Telemetry follows the training builders (generate.build_request / flows.nf_features), so the model sees the same
representation it was trained on. A record over the 2048-token budget is not scored (reported as over_budget).
Training overlap: a benchmark payload found verbatim in the v10 train split is flagged in_train; results are
reported for all records and for the clean subset.

usage: python -m eval.public_bench build NAME   -> reports/benchmarks/NAME.jsonl
       python -m eval.public_bench score NAME   -> NAME.results.jsonl via the SageMaker endpoint (resumable)
       python -m eval.public_bench report       -> reports/benchmarks_v10.json (+ SVM and CRS PL1 baselines)
"""
import argparse
import base64
import glob
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, unquote_plus, urlsplit

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT.parent / "raw_cache"
OUT = ROOT / "reports" / "benchmarks"
sys.path.insert(0, str(ROOT))
from data_pipeline import generate as G  # noqa: E402
from training import prepare as P  # noqa: E402

ATTACK = "attack"  # expected label for an attack outside the taxonomy (any attack verdict counts as detection)
GTW_TYPES = {"SQL Injection": ["sqli_classic", "sqli_obfuscated"], "XSS": ["xss_reflected_stored", "xss_dom_polyglot"],
             "RCE": ["rce_cmd_injection"], "Shell": ["rce_cmd_injection"], "Path Traversal": ["path_traversal_lfi"],
             "LFI": ["path_traversal_lfi"], "LDAP Injection": ["ldap_xpath_injection"], "NoSQL Injection": ["nosqli"],
             "SST Injection": ["ssti"], "CRLF Injection": ["crlf_header_injection"], "XXE": ["xxe"]}
OAS_TYPES = {"sqli": GTW_TYPES["SQL Injection"], "xss": GTW_TYPES["XSS"], "cmdexe": ["rce_cmd_injection"],
             "traversal": ["path_traversal_lfi"], "xxe": ["xxe"], "log4shell": ATTACK, "shellshock": ATTACK}
UNSW_TYPES = {"Reconnaissance": ["port_scan_recon"],
              "DoS": ["syn_flood_ddos", "udp_amplification", "slowloris_slowhttp", "http2_rapid_reset_continuation"]}
SOAP = ('<?xml version="1.0" encoding="UTF-8"?>\n<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"\n'
        '\txmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        '\t<soapenv:Header>\n\t\t<ns1:RequestHeader soapenv:actor="http://schemas.xmlsoap.org/soap/actor/next"\n'
        '\t\t\tsoapenv:mustUnderstand="0" xmlns:ns1="https://www.google.com/apis/ads/publisher/v202002">\n'
        '\t\t</ns1:RequestHeader>\n\t</soapenv:Header>\n\t<soapenv:Body>\n'
        '\t\t<getAdUnitsByStatement xmlns="https://www.google.com/apis/ads/publisher/v202002">\n'
        '\t\t\t<filterStatement>\n\t\t\t\t<{p}>{v}</{p}>\n\t\t\t</filterStatement>\n\t\t</getAdUnitsByStatement>\n'
        '\t</soapenv:Body>\n</soapenv:Envelope>')


def rec(rid, bench, category, expected, tel, payload=None, **meta):
    return {"id": rid, "bench": bench, "category": category, "expected": expected, "payload": payload,
            "telemetry": tel, **meta}


# ---- GoTestWAF ----------------------------------------------------------------------------------------------
def _gtw_encode(enc, s):
    if enc == "URL":
        return quote(s, safe="$&+:=@")  # Go url.PathEscape
    if enc == "Base64Flat":
        return base64.b64encode(s.encode()).decode().replace("=", "")
    if enc == "Base64":
        return base64.b64encode(s.encode()).decode()
    return s


def _go_xml_escape(s):  # encoding/xml EscapeText as xml.Encoder writes a string: <string>...</string>
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&#34;"), ("'", "&#39;"),
                 ("\t", "&#x9;"), ("\n", "&#xA;"), ("\r", "&#xD;")):
        s = s.replace(a, b)
    return f"<string>{s}</string>"


def _gtw_request(r, ph, enc, payload):
    """GoTestWAF placeholder -> telemetry. `wire` is what GoTestWAF puts on the wire. build_request keeps '%' in
    query/path values as already-encoded (attacker literal), so those get the wire string and the server's decoding
    follows from it; the form builder escapes everything, so it gets the server-decoded value. Multipart, JSON and
    headers are not URL-decoded by the server: the wire string is the value."""
    wire = _gtw_encode(enc, payload)
    param = "%010x" % r.getrandbits(40)  # GoTestWAF RandomHex(5)
    if ph == "URLParam":
        return G.build_request(r, "query", param, wire, [])
    if ph == "URLPath":
        return G.build_request(r, "path", param, wire, [])
    if ph == "HTMLForm":
        return G.build_request(r, "form", param, unquote_plus(wire), [])
    if ph == "HTMLMultipartForm":
        return G.build_request(r, "multipart", param, wire, [])
    if ph == "JSONRequest":
        return G.build_request(r, "json", param, wire, [], layer="json_unicode")
    if ph == "XMLBody":
        return G.build_request(r, "xml", param, wire, [])
    if ph == "SOAPBody":
        tel = G.build_request(r, "xml", param, SOAP.format(p=param, v=_go_xml_escape(wire)), [])
        tel["http"]["headers"].append(["SOAPAction", '"http://schemas.xmlsoap.org/soap/actor/next"'])
        return tel
    if ph in ("Header", "UserAgent"):
        tel = G.build_request(r, "query", "page", "1", [])
        h = tel["http"]["headers"]
        if ph == "UserAgent":
            h[:] = [kv for kv in h if kv[0].lower() != "user-agent"] + [["User-Agent", wire]]
        else:
            h.append(["X-" + param, wire])
        return tel
    return None  # RawRequest, GraphQL, gRPC: not converted


def build_gotestwaf():
    base = RAW / "gotestwaf" / "testcases"
    out, skipped = [], Counter()
    for f in sorted(base.glob("*/*.yml")):
        text = f.read_text()
        try:
            docs = [d for d in yaml.safe_load_all(text) if d]
        except yaml.YAMLError:  # Go's YAML accepts \' inside double quotes; Python's does not
            docs = [d for d in yaml.safe_load_all(text.replace("\\'", "'")) if d]
        for d in docs:
            ttype = d.get("type") or "false-pos"
            benign = f.parent.name == "false-pos"
            base_t = ttype.split(" ", 1)[1] if ttype.split(" ", 1)[0].endswith("KB") else ttype
            expected = "benign" if benign else GTW_TYPES.get(base_t, ATTACK)
            for i, payload in enumerate(d["payload"]):
                for enc in d["encoder"]:
                    for ph in d["placeholder"]:
                        ph = ph if isinstance(ph, str) else next(iter(ph))
                        rid = f"gtw:{f.parent.name}/{f.stem}:{i}:{enc}:{ph}"
                        tel = _gtw_request(G.rng_for(rid), ph, enc, str(payload))
                        if tel is None:
                            skipped[ph] += 1
                            continue
                        out.append(rec(rid, "gotestwaf", ttype, expected, tel, str(payload), encoder=enc,
                                       placeholder=ph, set=f.parent.name))
    return out, {"skipped_placeholders": dict(skipped)}


# ---- open-appsec ---------------------------------------------------------------------------------------------
def build_openappsec(n_legit=10_000, per_attack=800, seed=11):
    base = RAW / "openappsec_wcp"
    rnd, out, rep = random.Random(seed), [], {"legit_files": 0, "legit_requests": 0, "malicious": {}}
    with zipfile.ZipFile(base / "malicious.zip") as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".json")):
            cat = Path(name).stem
            reqs = json.loads(z.read(name))
            rep["malicious"][cat] = len(reqs)
            for j in sorted(rnd.sample(range(len(reqs)), min(per_attack, len(reqs)))):
                q = reqs[j]
                rid = f"oas:mal:{cat}:{j}"
                tel = G.telemetry_from_request(G.rng_for(rid), q["method"], q["url"], q.get("headers") or {}, q.get("data") or "")
                pay = unquote_plus(urlsplit(q["url"]).query.partition("=")[2]) or (q.get("data") or "")
                out.append(rec(rid, "openappsec", cat, OAS_TYPES.get(cat, ATTACK), tel, pay))
    with zipfile.ZipFile(base / "legitimate.zip") as z:
        files = sorted(n for n in z.namelist() if n.endswith(".json"))
        rep["legit_files"] = len(files)
        per = -(-n_legit // len(files))
        for name in files:
            reqs = json.loads(z.read(name))
            rep["legit_requests"] += len(reqs)
            for j in sorted(rnd.sample(range(len(reqs)), min(per, len(reqs)))):
                q = reqs[j]
                rid = f"oas:legit:{Path(name).stem}:{j}"
                tel = G.telemetry_from_request(G.rng_for(rid), q["method"], q["url"], q.get("headers") or {}, q.get("data") or "")
                out.append(rec(rid, "openappsec", "legitimate", "benign", tel, site=Path(name).stem))
    return out, rep


# ---- NF-UNSW-NB15 v1 -----------------------------------------------------------------------------------------
def build_unsw(n_benign=4000, per_attack=500, seed=11):
    import pandas as pd
    from data_pipeline.flows import nf_features, telemetry
    p = next((RAW / "nf_unsw_nb15_v1").rglob("NF-UNSW-NB15.csv"))
    df = pd.read_csv(p)
    df = df[df.Attack != "Attack"]  # one stray header-like row
    rep = {"rows": len(df), "by_label": df.Attack.value_counts().to_dict(), "rejected": Counter()}
    picks = [df[df.Attack == "Benign"].sample(n=n_benign, random_state=seed)]
    for cat, d in df[df.Attack != "Benign"].groupby("Attack"):
        picks.append(d.sample(n=min(per_attack, len(d)), random_state=seed))
    out = []
    for idx, row in pd.concat(picks).iterrows():
        f, why = nf_features(row)
        if why:
            rep["rejected"][why] += 1
            continue
        cat = row["Attack"]
        out.append(rec(f"unsw:{idx}", "unsw", cat, "benign" if cat == "Benign" else UNSW_TYPES.get(cat, ATTACK),
                       telemetry(f)))
    rep["rejected"] = dict(rep["rejected"])
    return out, rep


BUILDERS = {"gotestwaf": build_gotestwaf, "openappsec": build_openappsec, "unsw": build_unsw}


# ---- build: token budget + training overlap -------------------------------------------------------------------
def n_tokens(tok, ctrl, tel):
    msgs = [{"role": "system", "content": P.system_prompt()}, {"role": "user", "content": P.serialize_telemetry(P.add_decoded_views(tel), ctrl)}]
    return len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                   add_special_tokens=False)["input_ids"])


def build(name):
    recs, rep = BUILDERS[name]()
    tok = P.load_tokenizer()
    ctrl = P.control_strings(tok)
    train = (ROOT / "datasets_ready/fast_train.jsonl").read_text()
    rep["fitted"] = rep["unfittable"] = 0
    for r in recs:
        r["tokens"] = n_tokens(tok, ctrl, r["telemetry"])
        r["fitted"] = False
        if r["tokens"] > P.MAX_TOKENS and r["telemetry"].get("http") and name == "openappsec":
            t = P.fit_budget(r["telemetry"], lambda x: n_tokens(tok, ctrl, x))  # the sensor contract
            if t is not None:  # GoTestWAF's padded cases stay unscored: they test exactly this evasion
                r["telemetry"], r["tokens_original"], r["tokens"], r["fitted"] = t, r["tokens"], n_tokens(tok, ctrl, t), True
                rep["fitted"] += 1
            else:
                rep["unfittable"] += 1
        pay = r.get("payload") or ""
        r["in_train"] = bool(len(pay) >= 8 and r["expected"] != "benign"
                             and (json.dumps(pay, ensure_ascii=False)[1:-1] in train))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    rep |= {"records": len(recs), "over_budget": sum(r["tokens"] > P.MAX_TOKENS for r in recs),
            "in_train": sum(r["in_train"] for r in recs), "by_expected": Counter(
                "benign" if r["expected"] == "benign" else "attack" for r in recs)}
    (OUT / f"{name}.build.json").write_text(json.dumps(rep, indent=1, default=dict))
    print(json.dumps(rep, indent=1, default=dict))


# ---- score through the endpoint -------------------------------------------------------------------------------
def score(name, batch=12, workers=3):
    from deploy.client import invoke
    recs = [json.loads(l) for l in open(OUT / f"{name}.jsonl")]
    res_p = OUT / f"{name}.results.jsonl"
    done = {json.loads(l)["id"] for l in open(res_p)} if res_p.exists() else set()
    todo = [r for r in recs if r["id"] not in done and r["tokens"] <= P.MAX_TOKENS]
    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    print(f"{name}: {len(done)} done, {len(todo)} to score in {len(chunks)} calls", flush=True)

    def run(chunk):
        for attempt in range(3):
            try:
                return chunk, invoke([r["telemetry"] for r in chunk])
            except Exception as e:  # throttling / timeout: retry, then record the failure
                err = f"{type(e).__name__}: {e}"[:300]
        return chunk, [{"error": err}] * len(chunk)
    with open(res_p, "a") as fh, ThreadPoolExecutor(workers) as ex:
        for n, (chunk, res) in enumerate(ex.map(run, chunks)):
            for r, x in zip(chunk, res):
                fh.write(json.dumps({"id": r["id"], **x}) + "\n")
            fh.flush()
            if n % 25 == 0:
                print(f"  {n + 1}/{len(chunks)} calls", flush=True)


# ---- report ----------------------------------------------------------------------------------------------------
def _metrics(rows, classes_benign):
    """rows: (record, result). Attack: blocked (verdict BLOCK), flagged (top class is an attack), correct class
    (top in expected), abstained. Benign: false block, flagged, abstained."""
    m = defaultdict(int)
    for r, x in rows:
        d = x.get("decision") or {}
        top = (x.get("top") or [[None]])[0][0]
        atk_top = top is not None and top not in classes_benign
        if r["expected"] == "benign":
            m["benign_n"] += 1
            m["false_block"] += d.get("verdict") == "BLOCK"
            m["benign_flagged"] += atk_top
            m["benign_abstain"] += d.get("verdict") == "ABSTAIN"
        else:
            m["attack_n"] += 1
            m["blocked"] += d.get("verdict") == "BLOCK"
            m["flagged"] += atk_top
            m["abstained"] += d.get("verdict") == "ABSTAIN"
            if isinstance(r["expected"], list):
                m["class_n"] += 1
                m["class_ok"] += top in r["expected"]
    out = {}
    if m["attack_n"]:
        out |= {"attack_n": m["attack_n"], "block_rate": round(m["blocked"] / m["attack_n"], 4),
                "detect_rate": round(m["flagged"] / m["attack_n"], 4),
                "attack_abstain_rate": round(m["abstained"] / m["attack_n"], 4)}
        if m["class_n"]:
            out["class_accuracy"] = round(m["class_ok"] / m["class_n"], 4)
    if m["benign_n"]:
        out |= {"benign_n": m["benign_n"], "false_block_rate": round(m["false_block"] / m["benign_n"], 4),
                "benign_flagged_rate": round(m["benign_flagged"] / m["benign_n"], 4),
                "benign_abstain_rate": round(m["benign_abstain"] / m["benign_n"], 4)}
    return out


def _baselines(recs, benign):
    """SVM (char TF-IDF, trained on the v10 train split) and CRS PL1 on the same serialized telemetry."""
    from eval import baselines as B
    tok = P.load_tokenizer()
    ctrl = P.control_strings(tok)
    rows = B.load("fast_")
    bench = [(P.serialize_telemetry(r["telemetry"], ctrl),
              "benign_standard_traffic" if r["expected"] == "benign" else "rce_cmd_injection") for r in recs]
    rows = {"train": rows["train"], "test": bench, "test_unseen": []}
    svm, crs = B.tfidf_svm(rows, benign).get("test", {}), B.crs_pl1(rows, benign).get("test", {})
    keep = ("benign_fpr", "attack_miss_rate", "attack_vs_benign_accuracy", "n", "n_http")
    return {"tfidf_svm": {k: v for k, v in svm.items() if k in keep}, "crs_pl1": {k: v for k, v in crs.items() if k in keep}}


def report():
    tax = yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    benign = {k for k, v in tax.items() if v["is_benign"]}
    full = {}
    for name in BUILDERS:
        if not (OUT / f"{name}.results.jsonl").exists():
            continue
        recs = {r["id"]: r for r in map(json.loads, open(OUT / f"{name}.jsonl"))}
        res = {x["id"]: x for x in map(json.loads, open(OUT / f"{name}.results.jsonl"))}
        rows = [(recs[i], x) for i, x in res.items() if x.get("top")]  # scored (errors carry no top classes)
        lat = sorted(x.get("latency_s", 0) for _, x in rows)
        b = {"build": json.load(open(OUT / f"{name}.build.json")), "scored": len(rows),
             "errors": sum("error" in x for x in res.values()),
             "latency_s": {"median": lat[len(lat) // 2], "p90": lat[int(len(lat) * .9)]} if lat else None,
             "all": _metrics(rows, benign), "clean": _metrics([(r, x) for r, x in rows if not r["in_train"]], benign),
             "by_category": {c: _metrics([(r, x) for r, x in rows if r["category"] == c], benign)
                             for c in sorted({r["category"] for r, _ in rows})}}
        if name == "openappsec":  # requests shortened to the token budget vs untouched ones
            b["by_fitted"] = {str(v): _metrics([(r, x) for r, x in rows if r.get("fitted") == v], benign) for v in (False, True)}
        if name == "gotestwaf":
            for key in ("encoder", "placeholder"):
                b[f"by_{key}"] = {v: _metrics([(r, x) for r, x in rows if r[key] == v and r["expected"] != "benign"], benign)
                                  for v in sorted({r[key] for r, _ in rows})}
        b["baselines_same_records"] = _baselines([r for r, _ in rows], benign)
        full[name] = b
    (ROOT / "reports/benchmarks_v10.json").write_text(json.dumps(full, indent=1))
    print(json.dumps({k: {"all": v["all"], "clean": v["clean"], "baselines": v["baselines_same_records"]}
                      for k, v in full.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score", "report"])
    ap.add_argument("name", nargs="?")
    a = ap.parse_args()
    {"build": lambda: build(a.name), "score": lambda: score(a.name), "report": report}[a.cmd]()

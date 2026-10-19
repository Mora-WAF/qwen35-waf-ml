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
       python -m eval.public_bench report [--tag T] [--no-baselines] -> reports/benchmarks_T.json (+ SVM, CRS PL1)
  Other models (19 Oct; deploy/run_benchmarks.sh runs the whole sequence):
       pack [NAME...]                      every record -> bench_in.jsonl.gz (one Colab run: colab/vllm_bench.py fp)
       unpack FP_OUT --tag T               -> NAME.T.results.jsonl
       retag TRAIN.jsonl --key in_train_v11  training overlap against another build
       redecide --src T --policy P --as T2 re-apply a frozen policy offline (exact with probs)
       gemma --scores S --policy P --as T2 Gemma TPU window scores -> results (Gemma's frozen policy)
       ensemble --policy P --members A,B --as T2   combine member distributions (eval.policy.ensemble_probs)
       compare --tags A,B,...              per benchmark / category on shared records -> reports/benchmarks_compare.json
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
             "SST Injection": ["ssti"], "CRLF Injection": ["crlf_header_injection"], "XXE": ["xxe"],
             "Mail Injection": ["crlf_header_injection"]}  # 18 Oct: mail-protocol injection mapped to crlf (user decision)
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


def report(tag="v10", baselines=True):
    tax = yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    benign = {k for k, v in tax.items() if v["is_benign"]}
    full = {}
    for name in BUILDERS:
        if not any(f.exists() for f in results_files(name, tag)):
            continue
        recs = load_records(name)
        res = load_results(name, tag)
        rows = [(recs[i], x) for i, x in res.items() if x.get("top")]  # scored (errors carry no top classes)
        lat = sorted(x.get("latency_s", 0) for _, x in rows)
        b = {"build": json.load(open(OUT / f"{name}.build.json")), "scored": len(rows),
             "errors": sum("error" in x for x in res.values()),
             "latency_s": {"median": lat[len(lat) // 2], "p90": lat[int(len(lat) * .9)]} if lat else None,
             "all": _metrics(rows, benign), "clean": _metrics([(r, x) for r, x in rows if _clean(r)], benign),
             "by_category": {c: _metrics([(r, x) for r, x in rows if r["category"] == c], benign)
                             for c in sorted({r["category"] for r, _ in rows})}}
        if name == "openappsec":  # requests shortened to the token budget vs untouched ones
            b["by_fitted"] = {str(v): _metrics([(r, x) for r, x in rows if r.get("fitted") == v], benign) for v in (False, True)}
        if name == "gotestwaf":
            for key in ("encoder", "placeholder"):
                b[f"by_{key}"] = {v: _metrics([(r, x) for r, x in rows if r[key] == v and r["expected"] != "benign"], benign)
                                  for v in sorted({r[key] for r, _ in rows})}
        if baselines:  # needs the tokenizer and the build's train split
            b["baselines_same_records"] = _baselines([r for r, _ in rows], benign)
        full[name] = b
    (ROOT / f"reports/benchmarks_{tag}.json").write_text(json.dumps(full, indent=1))
    print(json.dumps({k: {"all": v["all"], "clean": v["clean"], "baselines": v.get("baselines_same_records")}
                      for k, v in full.items()}, indent=1))


# ---- per-model results, offline re-decisions, ensemble, comparison (19 Oct) ---------------------------------------
# Every model's benchmark results live in NAME.TAG.results.jsonl (v10, the first endpoint run, keeps its original
# files). A result carries the decision; results scored with return_probs also carry the calibrated distribution of
# the deciding window ("probs") and its temperature, so a policy change (envelope resolution, decision view) or an
# ensemble is re-decided offline through eval.policy.decide_from_probs, the endpoint's own decision path.
PREFIX = {"gtw:": "gotestwaf", "oas:": "openappsec", "unsw:": "unsw"}
LEGACY = {"v10": {"gotestwaf": ["gotestwaf.results.jsonl"], "unsw": ["unsw.results.jsonl"],
                  "openappsec": ["openappsec.results.jsonl", "openappsec_legit_v10.results.jsonl"]}}


def results_files(name, tag):
    return [OUT / f for f in LEGACY.get(tag, {}).get(name, [f"{name}.{tag}.results.jsonl"])]


def load_results(name, tag):
    res = {}
    for f in results_files(name, tag):
        if f.exists():
            res |= {x["id"]: x for x in map(json.loads, open(f))}
    return res


def load_records(name):
    return {r["id"]: r for r in map(json.loads, open(OUT / f"{name}.jsonl"))}


def _write_results(name, tag, rows):
    f = OUT / f"{name}.{tag}.results.jsonl"
    assert tag not in LEGACY, f"{tag} results are the original run's files"
    f.write_text("".join(json.dumps(x) + "\n" for x in rows))
    return f


def pack(names, out):
    """One scoring input for every record of the named benchmarks ({id, telemetry}, gzip): one Colab VM, one merge,
    one pass (colab/vllm_bench.py fp). Over-budget records included: the endpoint's sensor contract fits or windows
    them itself, or answers over_budget."""
    import gzip
    n = 0
    with gzip.open(out, "wt") as fh:
        for name in names:
            for r in load_records(name).values():
                fh.write(json.dumps({"id": r["id"], "telemetry": r["telemetry"]}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} records -> {out} ({Path(out).stat().st_size / 1e6:.1f} MB; Colab uploads above ~10 MB fail)")


def unpack(results, tag):
    """Split a fp_out.jsonl (colab/vllm_bench.py fp) into NAME.TAG.results.jsonl by id prefix. Never overwrites."""
    by = defaultdict(list)
    for x in map(json.loads, open(results)):
        name = next((v for k, v in PREFIX.items() if x["id"].startswith(k)), None)
        assert name, f"unknown benchmark id {x['id']}"
        by[name].append(x)
    for name, rows in by.items():
        f = OUT / f"{name}.{tag}.results.jsonl"
        assert not f.exists(), f"{f} exists: raw results are kept, pick another tag"
        _write_results(name, tag, rows)
        print(f"{name}: {len(rows)} -> {f}")


def retag(train, key):
    """Recompute the training-overlap flag against another build's train split (e.g. the v11 build: key in_train_v11).
    Report and compare drop a record from the clean subset when any in_train* flag is set."""
    text = Path(train).read_text()
    for name in BUILDERS:
        fp = OUT / f"{name}.jsonl"
        if not fp.exists():
            continue
        recs = [json.loads(l) for l in open(fp)]
        for r in recs:
            pay = r.get("payload") or ""
            r[key] = bool(len(pay) >= 8 and r["expected"] != "benign" and json.dumps(pay, ensure_ascii=False)[1:-1] in text)
        fp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        print(f"{name}: {sum(r[key] for r in recs)} of {len(recs)} {key}")


def _probs_for(x, T):
    """A result's calibrated distribution at temperature T (re-tempered when it was scored at another one:
    log p = s / T0 + c, so softmax(log p * T0 / T) is the distribution at T), or None."""
    import numpy as np
    if not x.get("probs"):
        return None
    p = np.asarray(x["probs"], dtype=np.float64)
    T0 = float(x.get("temperature") or T)
    if abs(T0 - T) < 1e-9:
        return p
    z = np.log(np.maximum(p, 1e-300)) * T0 / T
    e = np.exp(z - z.max())
    return e / e.sum()


def _passthrough(x):
    """Results without a distribution to re-decide: over-budget decisions, errors."""
    return {k: v for k, v in x.items() if k in ("id", "decision", "diagnostics", "error", "truncated", "windows")}


def redecide(name, src, policy, dst, scored_temperature=None):
    """Re-apply a frozen policy to a model's results. With probs: exact for every decision view. Without (v10): the
    top_class view from the top class and its probability, exact only at the temperature the run scored with
    (scored_temperature, checked against the policy)."""
    from eval import policy as POL
    tax, _, _ = POL.load()
    pol = yaml.safe_load(open(ROOT / policy))
    T = float(pol["temperature"])
    recs, out, n = load_records(name), [], Counter()
    for i, x in load_results(name, src).items():
        tel = recs[i]["telemetry"]
        p = _probs_for(x, T)
        if p is not None:
            d, diag, top = POL.decide_from_probs(p, tel, tax, pol)
            y = {"id": i, "decision": d, "diagnostics": diag, "top": top, "probs": [round(float(v), 7) for v in p], "temperature": T}
        elif x.get("top") and (x.get("diagnostics") or {}).get("reason") != "over_budget":
            assert pol.get("act_on", "top_class") == "top_class", "attack_mass needs the full distribution (probs)"
            assert scored_temperature is not None and abs(float(scored_temperature) - T) < 1e-6, \
                "results without probs: give --scored-temperature, equal to the policy's temperature"
            cls, conf = x["top"][0]
            raw = json.dumps({"threat_type": cls}, separators=(",", ":"))
            d, diag = POL.decide(raw, tel.get("mode"), tel.get("context"), float(conf), tax, pol,
                                 (tel.get("flow") or {}).get("features"), (tel.get("http") or {}).get("headers"))
            y = {"id": i, "decision": d, "diagnostics": diag, "top": x["top"]}
        else:
            y = _passthrough(x)
        n[((x.get("decision") or {}).get("verdict"), (y.get("decision") or {}).get("verdict"))] += 1
        out.append({**y, "redecided_from": src, "policy": policy})
    f = _write_results(name, dst, out)
    print(f"{name}: {len(out)} -> {f}; verdict changes {dict((k, v) for k, v in n.items() if k[0] != k[1])}")


def gemma_results(name, scores, manifest, policy, dst):
    """Gemma TPU scores of the benchmark windows (training.gemma_tpu.data bench -> train_tunix.py score bench) ->
    NAME.DST.results.jsonl, exactly as the endpoint decides: calibrated distribution per window, the window with the
    largest attack mass decides, decide_from_probs with Gemma's frozen policy; over-budget records -> over_budget."""
    import numpy as np
    from eval import calibration as CAL
    from eval import policy as POL
    tax, _, _ = POL.load()
    pol = yaml.safe_load(open(ROOT / policy))
    T = float(pol["temperature"])
    benign = np.array([v["is_benign"] for v in tax["classes"].values()])
    man = json.load(open(manifest))
    win = defaultdict(list)
    f = open(scores)
    assert json.loads(f.readline())["classes"] == list(tax["classes"])
    for x in map(json.loads, f):
        rid, _, k = x["id"].rpartition("#")
        win[rid].append(CAL.softmax(x["scores"], T))
    recs, out = load_records(name), []
    for rid, m in man["records"].items():
        if not rid.startswith(next(k for k, v in PREFIX.items() if v == name)):
            continue
        if m["windows"] == 0:
            out.append({"id": rid, "decision": POL.over_budget_decision(pol), "diagnostics": {"reason": "over_budget"},
                        "truncated": True, "windows": 0})
            continue
        assert len(win[rid]) == m["windows"], f"{rid}: {len(win[rid])} of {m['windows']} windows scored"
        p = max(win[rid], key=lambda q: 1.0 - float(q[benign].sum()))
        d, diag, top = POL.decide_from_probs(p, recs[rid]["telemetry"], tax, pol)
        out.append({"id": rid, "decision": d, "diagnostics": diag, "top": top, "windows": m["windows"],
                    "truncated": m["truncated"], "probs": [round(float(v), 7) for v in p], "temperature": T})
    f = _write_results(name, dst, out)
    print(f"{name}: {len(out)} -> {f}")


def ensemble(name, policy, members, dst):
    """Combine member results (tags, in the ensemble policy's member order) with eval.policy.ensemble_probs and decide
    with the ensemble's frozen policy. A record any member could not fit is over_budget; a member error stays one."""
    from eval import policy as POL
    tax, _, _ = POL.load()
    pol = yaml.safe_load(open(ROOT / policy))
    spec = pol["ensemble"]["members"]
    assert len(members) == len(spec), f"{len(spec)} members in {policy}"
    res = [load_results(name, t) for t in members]
    recs, out = load_records(name), []
    for i in sorted(set.intersection(*(set(r) for r in res))):
        xs = [r[i] for r in res]
        ps = [_probs_for(x, float(m["temperature"])) for x, m in zip(xs, spec)]
        if all(p is not None for p in ps):
            p = POL.ensemble_probs(ps, pol)
            d, diag, top = POL.decide_from_probs(p, recs[i]["telemetry"], tax, pol)
            out.append({"id": i, "decision": d, "diagnostics": diag, "top": top, "probs": [round(float(v), 7) for v in p],
                        "temperature": float(pol["temperature"])})
        elif any((x.get("diagnostics") or {}).get("reason") == "over_budget" for x in xs):
            out.append({"id": i, "decision": POL.over_budget_decision(pol), "diagnostics": {"reason": "over_budget"}})
        else:
            out.append({"id": i, "decision": None, "error": "member result without a distribution: " +
                        ", ".join(t for t, x in zip(members, xs) if not x.get("probs"))})
    f = _write_results(name, dst, out)
    print(f"{name}: {len(out)} -> {f} (members {members})")


def _clean(r):
    return not any(v for k, v in r.items() if k.startswith("in_train"))


def compare(tags, out="reports/benchmarks_compare.json"):
    """Per benchmark and category, every tag on the records all tags decided (shared eligible inputs, spec §10),
    plus each tag on all its records. Metrics as in report (block / detect / abstain / false block)."""
    tax = yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    benign = {k for k, v in tax.items() if v["is_benign"]}
    full = {}
    for name in BUILDERS:
        if not (OUT / f"{name}.jsonl").exists():
            continue
        recs = load_records(name)
        res = {t: {i: x for i, x in load_results(name, t).items() if x.get("decision")} for t in tags}
        if not all(res.values()):
            print(f"{name}: no decided results for {[t for t in tags if not res[t]]}, skipped")
            continue
        shared = sorted(set.intersection(*(set(r) for r in res.values())))
        b = {"records": len(recs), "decided": {t: len(r) for t, r in res.items()}, "shared": len(shared),
             "shared_all": {}, "shared_clean": {}, "by_category": {}, "all": {}}
        for t in tags:
            rows = [(recs[i], res[t][i]) for i in shared]
            b["shared_all"][t] = _metrics(rows, benign)
            b["shared_clean"][t] = _metrics([(r, x) for r, x in rows if _clean(r)], benign)
            b["all"][t] = _metrics([(recs[i], x) for i, x in res[t].items()], benign)
        for c in sorted({recs[i]["category"] for i in shared}):
            b["by_category"][c] = {t: _metrics([(recs[i], res[t][i]) for i in shared if recs[i]["category"] == c], benign)
                                   for t in tags}
        full[name] = b
    (ROOT / out).write_text(json.dumps(full, indent=1))
    for name, b in full.items():
        print(f"== {name}: shared {b['shared']} of {b['records']} records")
        for t in tags:
            m = b["shared_all"][t]
            print(f"  {t:14s} " + " ".join(f"{k}={v}" for k, v in m.items() if not k.endswith("_n")))
    print("->", out)
    return full


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score", "report", "pack", "unpack", "retag", "redecide", "gemma",
                                    "ensemble", "compare"])
    ap.add_argument("name", nargs="*", help="benchmark name(s); unpack: the fp_out.jsonl; retag: the train jsonl")
    ap.add_argument("--tag", default="v10", help="report: results tag; unpack: tag to write")
    ap.add_argument("--src", help="redecide: results tag to re-decide")
    ap.add_argument("--as", dest="dst", help="redecide / gemma / ensemble: tag to write")
    ap.add_argument("--policy", help="redecide / gemma / ensemble: frozen policy file")
    ap.add_argument("--scored-temperature", type=float, help="redecide without probs: the run's scoring temperature")
    ap.add_argument("--members", help="ensemble: member tags in the policy's order, comma-separated")
    ap.add_argument("--scores", help="gemma: SCORES_bench.jsonl from the TPU")
    ap.add_argument("--manifest", default="reports/gemma_tpu/data_bench/bench_manifest.json", help="gemma: export manifest")
    ap.add_argument("--tags", default="v10,v11", help="compare: tags, comma-separated")
    ap.add_argument("--compare-out", default="reports/benchmarks_compare.json", help="compare: output")
    ap.add_argument("--key", default="in_train_v11", help="retag: flag name")
    ap.add_argument("--out", default="reports/benchmarks/bench_in.jsonl.gz", help="pack: output")
    ap.add_argument("--no-baselines", action="store_true", help="report: skip the SVM / CRS baselines")
    a = ap.parse_args()
    names = a.name or [n for n in BUILDERS if (OUT / f"{n}.jsonl").exists()]
    each = lambda fn: [fn(n) for n in names]  # noqa: E731
    {"build": lambda: build(a.name[0]), "score": lambda: score(a.name[0]),
     "report": lambda: report(a.tag, not a.no_baselines),
     "pack": lambda: pack(names, a.out), "unpack": lambda: unpack(a.name[0], a.tag),
     "retag": lambda: retag(a.name[0], a.key),
     "redecide": lambda: each(lambda n: redecide(n, a.src, a.policy, a.dst, a.scored_temperature)),
     "gemma": lambda: each(lambda n: gemma_results(n, a.scores, a.manifest, a.policy, a.dst)),
     "ensemble": lambda: each(lambda n: ensemble(n, a.policy, a.members.split(","), a.dst)),
     "compare": lambda: compare(a.tags.split(","), a.compare_out)}[a.cmd]()

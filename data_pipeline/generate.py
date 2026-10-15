"""Deterministic generators: observed seeds + scenario oracles -> telemetry records (spec §6).

Three deterministic roles here (LLM-worker roles go in subagents/):
  - wrap_http:  place an observed attack seed into a generated HTTP request  -> provenance `hybrid`
  - synth_flow: derive aggregate L4 flow features from a bounded event scenario -> `synthetic`
  - synth_benign: ordinary and hard-negative benign telemetry                 -> `synthetic`

The scenario params ARE the label oracle (label_evidence.method == scenario_oracle). No string
is trusted as proof of exploitability. All randomness is seeded by record id -> reproducible.
"""
import hashlib
import json
import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "data" / "seeds"
SYS = None  # lazily loaded system prompt


def rng_for(key):
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def system_prompt():
    global SYS
    if SYS is None:
        from training.prepare import system_prompt as sp
        SYS = sp()
    return SYS


# ---- HTTP wrapping (seeded L7 attack classes) ------------------------------------------------
# Label-shortcut hostnames (spec §8): seeds say "evil.com"/"attacker.example"; a model would
# learn the hostname, not the mechanism. Replace with neutral domains that benign traffic uses too.
SHORTCUT_HOST = re.compile(r"[A-Za-z0-9.-]*(?:evil|attacker|malicious|hacker)[A-Za-z0-9-]*(?:\.[A-Za-z]{2,})+", re.I)
NEUTRAL_HOSTS = ["cdn-static.net", "files.partner-site.org", "img.example.org", "assets.shopfront.io",
                 "status.vendor-portal.com", "media.newsdesk.net"]


# Class-name tokens that appear in seeds by convention only (XXE entity names are arbitrary).
CLASS_TOKEN = re.compile(r"\bxxe\b", re.I)
NEUTRAL_ENTITIES = ["ent", "doc", "ext", "item", "ref"]


def neutralize_hosts(text, r):
    """Returns (text, changed). Deterministic given r."""
    out = SHORTCUT_HOST.sub(lambda m: r.choice(NEUTRAL_HOSTS), text)
    name = r.choice(NEUTRAL_ENTITIES)  # one consistent rename per payload keeps &ent; refs valid
    out = CLASS_TOKEN.sub(name, out)
    return out, out != text


# Shared request vocabulary (spec §8 counterbalancing): attack AND benign requests draw endpoint,
# host, header profile, locale and parameter names from the SAME pools, so none of them carries
# label signal. Only the parameter VALUE differs.
ENDPOINTS = ["/api/v1/search", "/search", "/products", "/products/view", "/user/profile", "/login",
             "/api/v1/orders", "/comments", "/upload", "/render", "/admin/report", "/redirect",
             "/download", "/api/v2/items", "/feedback", "/export", "/account/settings", "/docs/view"]
HOSTS = ["app.example.test", "shop.example.test", "api.example.test", "portal.example.test"]
PARAMS = ["q", "id", "name", "next", "file", "page", "sort", "lang", "template", "url", "ref",
          "filter", "user", "path", "query", "redirect_to"]
LOCALES = ["en-US,en;q=0.9", "de-DE,de;q=0.8", "ja-JP", "es-ES,es;q=0.9", "ar-SA", "fr-FR", "pt-BR", "zh-CN"]
HEADER_PROFILES = {
    "chrome": [["User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"],
               ["Accept", "text/html,application/xhtml+xml,*/*;q=0.8"], ["Sec-Fetch-Mode", "navigate"],
               ["Sec-Fetch-Site", "same-origin"], ["Accept-Encoding", "gzip, deflate, br"]],
    "firefox": [["User-Agent", "Mozilla/5.0 (X11; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0"],
                ["Accept", "text/html,application/xhtml+xml,*/*;q=0.8"], ["Accept-Encoding", "gzip, deflate, br"],
                ["Connection", "keep-alive"]],
    "safari_ios": [["User-Agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1"],
                   ["Accept", "*/*"], ["Accept-Encoding", "gzip, deflate, br"]],
    "curl": [["User-Agent", "curl/8.7.1"], ["Accept", "*/*"]],
    "python": [["User-Agent", "python-requests/2.32.3"], ["Accept-Encoding", "gzip, deflate"], ["Accept", "*/*"],
               ["Connection", "keep-alive"]],
    "okhttp": [["User-Agent", "okhttp/4.12.0"], ["Accept-Encoding", "gzip"], ["Connection", "Keep-Alive"]],
    "go": [["User-Agent", "Go-http-client/2.0"], ["Accept-Encoding", "gzip"]],
    "axios": [["User-Agent", "axios/1.7.7"], ["Accept", "application/json, text/plain, */*"],
              ["Accept-Encoding", "gzip, compress, deflate, br"]],
}
# allowed payload positions per class (where the mechanism can actually be delivered)
SINKS = {"sqli_classic": ["query", "form", "json", "cookie"], "sqli_obfuscated": ["query", "form", "json", "cookie"],
         "nosqli": ["json", "query", "form"], "xss_reflected_stored": ["query", "form", "json"],
         "xss_dom_polyglot": ["query", "form", "json"], "rce_cmd_injection": ["query", "form", "json"],
         "ssti": ["query", "form", "json"], "xxe": ["xml"], "crlf_header_injection": ["query", "path"],
         "ldap_xpath_injection": ["query", "form"], "path_traversal_lfi": ["query", "path"],
         "open_redirect_cors": ["query"]}
BENIGN_VALUES = {
    "q": ["running shoes", "Kaffeemaschine günstig", "東京 ホテル", "zapatos de mujer", "حقيبة ظهر", "O'Brien family history",
          "4k monitor 27\"", "tom & jerry dvd", "C++ primer 5th edition", "100% cotton t-shirt", "<3 valentines gifts"],
    "id": ["10423", "8812", "a7f3c9e1", "550e8400-e29b-41d4-a716-446655440000", "77"],
    "page": ["1", "2", "15"], "sort": ["price_asc", "-created_at", "name", "rating desc"],
    "lang": ["en", "de", "ja", "pt-BR"], "next": ["/account", "/cart?step=2", "/orders/88"],
    "file": ["report_2026-09.pdf", "invoice-8812.csv", "logo.png"], "user": ["mkoch", "j.tanaka", "ana.silva@example.test"],
    "filter": ["status:open", "category=books", "price<50"], "template": ["default", "compact", "print"],
}
BENIGN_XML = [
    '<?xml version="1.0" encoding="UTF-8"?><soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body><GetOrder><id>8812</id></GetOrder></soap:Body></soap:Envelope>',
    '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://shop.example.test/products/77</loc><lastmod>2026-09-30</lastmod></url></urlset>',
    '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>Release notes</title><item><title>v2.4 &amp; fixes</title></item></channel></rss>',
    '<order><customer>Ana Silva</customer><line sku="A-100" qty="2"/><note>Leave at door &lt;back&gt;</note></order>',
    '<?xml version="1.0" encoding="ISO-8859-1"?><config><lang>de</lang><theme>compact</theme></config>',
    '<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Status</title><entry><title>Maintenance 02:00 UTC</title></entry></feed>',
]
QUOTE_SAFE = "%/:~!$'()*,;@[]"  # an attacker's literal %XX is sent as-is; only invalid URL chars get escaped


def _headers(r, host):
    prof = HEADER_PROFILES[r.choice(sorted(HEADER_PROFILES))]
    return [["Host", host], *[list(h) for h in prof], ["Accept-Language", r.choice(LOCALES)]]


def _benign_pairs(r, k):
    keys = r.sample(sorted(BENIGN_VALUES), k)
    return [[key, r.choice(BENIGN_VALUES[key])] for key in keys]


def build_request(r, sink, param, value, extra):
    """One HTTP request with `value` placed at `sink`. raw target = what the client sent;
    query/body pairs = the server's single-pass decoding; decoded_views only for deeper layers."""
    from urllib.parse import quote, unquote, unquote_plus
    host, path = r.choice(HOSTS), r.choice(ENDPOINTS)
    headers = _headers(r, host)
    http = {"method": "GET", "target": path, "protocol": r.choice(["HTTP/1.1", "HTTP/1.1", "HTTP/2"]),
            "headers": headers, "query": None, "cookies": None, "body": None, "body_encoding": "none"}
    views, stage = [], "request_headers"
    enc = quote(value, safe=QUOTE_SAFE)
    pairs_raw = [[k, quote(v, safe=QUOTE_SAFE)] for k, v in extra]
    if sink in ("query", "path"):
        if sink == "path":
            http["target"] = f"{path.rstrip('/')}/{enc}"
            qs = pairs_raw
        else:
            qs = pairs_raw + [[param, enc]]
            r.shuffle(qs)
        if qs:
            http["target"] += "?" + "&".join(f"{k}={v}" for k, v in qs)
            http["query"] = [[k, unquote_plus(v)] for k, v in qs]
        if sink == "path":
            views.append({"field": "target.path", "decoder": "url", "depth": 1, "value": unquote(enc)})
    elif sink == "cookie":
        http["cookies"] = [["session", "%032x" % r.getrandbits(128)], [param, enc]]
        headers.append(["Cookie", "; ".join(f"{k}={v}" for k, v in http["cookies"])])
        http["cookies"] = [[k, unquote(v)] for k, v in http["cookies"]]
    else:
        http["method"] = r.choice(["POST", "POST", "PUT"])
        stage = "request_headers_body"
        if sink == "form":
            body_pairs = pairs_raw + [[param, quote(value, safe="")]]
            r.shuffle(body_pairs)
            http["body"] = "&".join(f"{k}={v}" for k, v in body_pairs)
            headers.append(["Content-Type", "application/x-www-form-urlencoded"])
            views.append({"field": f"body.{param}", "decoder": "form", "depth": 1, "value": value})
        elif sink == "json":
            obj = dict(extra)
            obj[param] = value
            http["body"] = json.dumps(obj, ensure_ascii=False)
            headers.append(["Content-Type", "application/json"])
        elif sink == "xml":
            http["body"] = value
            headers.append(["Content-Type", r.choice(["application/xml", "text/xml; charset=utf-8"])])
        http["body_encoding"] = "utf8"
        headers.append(["Content-Length", str(len(http["body"].encode()))])
    # second decoding layer (double-encoded input): record it as a view, never replace the original
    for k, v in (http["query"] or []):
        if "%" in v and unquote(v) != v:
            views.append({"field": f"query.{k}", "decoder": "url", "depth": 2, "value": unquote(v)})
    if views:
        http["decoded_views"] = views
    return {"schema_version": 1, "mode": "L7_HTTP", "decision_stage": stage,
            "observed_until": "request_received", "truncated": False, "http": http}


def wrap_http_raw(seed):
    """Smuggling fixture: preserve the raw request block verbatim (exact framing, duplicate
    headers). First request line -> method/target/protocol; its headers kept in order; the
    entire raw block stays in body. This is a fixture, not proof of a successful desync (§5)."""
    raw, _ = neutralize_hosts(seed["payload"], rng_for(seed["id"]))
    lines = raw.split("\n")
    first = lines[0].split(" ")
    method = first[0] if first else "POST"
    target = first[1] if len(first) > 1 else "/"
    proto = first[2] if len(first) > 2 else "HTTP/1.1"
    headers = []
    for ln in lines[1:]:
        if ln.strip() == "":
            break
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers.append([k.strip(), v.strip()])
    http = {"method": method, "target": target, "protocol": proto, "headers": headers,
            "query": None, "cookies": None, "body": raw, "body_encoding": "utf8"}
    tel = {"schema_version": 1, "mode": "L7_HTTP", "decision_stage": "request_headers_body",
           "observed_until": "request_received", "truncated": False, "http": http}
    return tel, "hybrid"


def wrap_http(seed):
    if seed["class"] == "http_request_smuggling":
        return wrap_http_raw(seed)
    r = rng_for(seed["id"])
    cls = seed["class"]
    p, _ = neutralize_hosts(seed["payload"], r)
    sink = r.choice(SINKS[cls])
    param = r.choice(["next", "url", "redirect_to", "ref"]) if cls == "open_redirect_cors" else r.choice(PARAMS)
    extra = _benign_pairs(r, r.choice([0, 0, 1, 2]))
    extra = [kv for kv in extra if kv[0] != param]
    return build_request(r, sink, param, p, extra), "hybrid"


# ---- Flow synthesis (L4 classes) -------------------------------------------------------------
# Each profile derives a consistent aggregate from a bounded scenario. Rates = counts/duration.
FLOW_PROFILES = {
    "syn_flood_ddos": dict(fwd_frac=(0.97, 1.0), ports=[80, 443], window="fixed_window", dur=(4, 6), pkts=(8000, 60000), proto="tcp",
                           syn_frac=(0.95, 1.0), bytes_per_pkt=(40, 60), dst_fanout=(1, 2)),
    "udp_amplification": dict(fwd_frac=(0.0, 0.05), ports=[53, 123, 1900, 11211], window="fixed_window", dur=(4, 6), pkts=(2000, 20000), proto="udp",
                              syn_frac=(0, 0), bytes_per_pkt=(800, 4000), dst_fanout=(1, 3)),
    "slowloris_slowhttp": dict(fwd_frac=(0.6, 0.8), ports=[80, 443, 8080], window="fixed_window", dur=(20, 60), pkts=(50, 400), proto="tcp",
                               syn_frac=(0.05, 0.2), bytes_per_pkt=(60, 200), dst_fanout=(1, 1)),
    "port_scan_recon": dict(fwd_frac=(0.5, 0.65), window="fixed_window", dur=(2, 8), pkts=(500, 5000), proto="tcp",
                            syn_frac=(0.8, 1.0), bytes_per_pkt=(40, 44), dst_fanout=(200, 2000)),
    "credential_stuffing_bruteforce": dict(fwd_frac=(0.45, 0.6), ports=[21, 22, 443, 3389], window="fixed_window", dur=(10, 60), pkts=(200, 3000),
                                           proto="tcp", syn_frac=(0.1, 0.3), bytes_per_pkt=(300, 900),
                                           dst_fanout=(1, 2)),
    "c2_beaconing_anomaly": dict(fwd_frac=(0.45, 0.6), ports=[443, 8443, 80], window="full_flow", dur=(300, 1800), pkts=(20, 120), proto="tcp",
                                 syn_frac=(0.02, 0.1), bytes_per_pkt=(200, 1200), dst_fanout=(1, 1),
                                 periodic=True),
    "dns_tunneling_exfil": dict(fwd_frac=(0.45, 0.55), ports=[53], window="full_flow", dur=(30, 300), pkts=(200, 3000), proto="udp",
                                syn_frac=(0, 0), bytes_per_pkt=(120, 300), dst_fanout=(1, 1), dns=True),
    "tls_ja4_ua_spoofing": dict(fwd_frac=(0.35, 0.55), ports=[443], window="full_flow", dur=(1, 30), pkts=(20, 400), proto="tcp",
                                syn_frac=(0.02, 0.08), bytes_per_pkt=(400, 1400), dst_fanout=(1, 1),
                                ja4=True),
    "benign_standard_traffic": dict(fwd_frac=(0.25, 0.6), ports=[443, 80, 53, 8080, 22], window="full_flow", dur=(1, 120), pkts=(10, 500), proto="tcp",
                                    syn_frac=(0.01, 0.05), bytes_per_pkt=(400, 1400), dst_fanout=(1, 2)),
    "benign_high_rate_legit": dict(fwd_frac=(0.3, 0.6), ports=[443, 80], window="fixed_window", dur=(4, 6), pkts=(2000, 15000), proto="tcp",
                                   syn_frac=(0.02, 0.08), bytes_per_pkt=(500, 1400), dst_fanout=(1, 3)),
}


def sig(x, n=6):
    """Round to n significant digits: keeps relative precision for low and high rates."""
    return float(f"{x:.{n}g}")


def synth_flow(cls, idx):
    r = rng_for(f"flow:{cls}:{idx}")
    pr = FLOW_PROFILES[cls]
    dur = round(r.uniform(*pr["dur"]), 3)
    pkts = r.randint(*pr["pkts"])
    bpp = r.randint(*pr["bytes_per_pkt"])
    total_bytes = pkts * bpp
    syn = int(pkts * r.uniform(*pr["syn_frac"]))
    # CORE vocabulary only (config/features.yaml) - same names as the observed CIC/NF records
    flags = (2 if syn else 0) | (16 if pkts - syn > 0 else 0)  # SYN, ACK
    fwd = max(1, min(pkts, round(pkts * r.uniform(*pr.get("fwd_frac", (0.4, 0.6))))))
    bwd = pkts - fwd
    spread = r.randint(0, max(1, bpp // 2))
    feats = {
        "duration_s": dur, "pkts_total": pkts, "bytes_total": total_bytes,
        "pkts_fwd": fwd, "pkts_bwd": bwd, "bytes_fwd": fwd * bpp, "bytes_bwd": bwd * bpp,
        "pkts_per_s": sig(pkts / dur), "bytes_per_s": sig(total_bytes / dur),
        "pkt_len_min": max(40, bpp - spread), "pkt_len_max": bpp + spread,
        "tcp_flags_bitmask": flags if pr["proto"] == "tcp" else 0,
        "ip_proto": 6 if pr["proto"] == "tcp" else 17,
        "dst_port": r.choice(pr["ports"]) if pr.get("ports") else r.randint(1, 65535),
    }
    if pr.get("periodic"):
        feats["iat_cv"] = round(r.uniform(0.0, 0.15), 4)  # low jitter -> beacon-like
    if pr.get("dns"):
        feats["qname_len_mean"] = round(r.uniform(40, 120), 2)
        feats["qname_entropy_bits_per_char"] = round(r.uniform(3.6, 4.5), 3)
    if pr.get("ja4"):
        feats["ja4_ua_consistent"] = 0.0  # measured mismatch; mapping justified in metadata
    tel = {"schema_version": 1, "mode": "L4_FLOW",
           "decision_stage": "flow_complete" if pr["window"] == "full_flow" else "flow_window",
           "observed_until": pr["window"], "truncated": False,
           "flow": {"window": {"kind": pr["window"], "seconds": None if pr["window"] == "full_flow" else dur},
                    "protocol": pr["proto"], "features": feats}}
    scenario = {"profile": cls, "duration_s": dur, "pkts": pkts, "bytes_per_pkt": bpp,
                "syn_fraction_range": pr["syn_frac"], "window_kind": pr["window"]}
    return tel, scenario


# ---- Benign HTTP (same endpoints / headers / params as attacks) ---------------------------------
def synth_benign_http(cls, idx):
    r = rng_for(f"benign:{cls}:{idx}")
    pairs = _benign_pairs(r, r.choice([1, 1, 2, 3]))
    param, value = pairs[-1]
    sink = r.choice(["query", "query", "form", "json", "cookie", "xml"])
    if sink == "xml":  # legitimate XML so the content type alone does not signal XXE
        value = r.choice(BENIGN_XML)
    tel = build_request(r, sink, param, value, pairs[:-1])
    if r.random() < 0.3:  # referrals from the same neutral hosts the attack seeds are rewritten to
        tel["http"]["headers"].append(["Referer", f"https://{r.choice(NEUTRAL_HOSTS)}/"])
    return tel, {"profile": cls, "sink": sink}


# ---- Record assembly -------------------------------------------------------------------------
def assemble(rec_id, cls, tel, provenance, evidence, meta_extra):
    from training.prepare import serialize_telemetry, control_strings, load_tokenizer
    global _CTRL
    try:
        ctrl = _CTRL
    except NameError:
        ctrl = _CTRL = control_strings(load_tokenizer())
    user = serialize_telemetry(tel, ctrl)
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps({"threat_type": cls}, separators=(",", ":"))}]
    meta = {"threat_type": cls, "provenance": provenance, "decision_stage": tel["decision_stage"],
            "label_evidence": evidence, "split": "unassigned", "token_len": None,
            "mechanism_tags": [], "template_family": meta_extra.get("template_family", cls),
            "dup_cluster": None, "generator": meta_extra.get("generator", "deterministic"),
            "source": meta_extra.get("source", "generated"),
            "source_revision": meta_extra.get("source_revision", "n/a"),
            "original_id": meta_extra.get("original_id", rec_id),
            "original_label": meta_extra.get("original_label", cls),
            "lineage_group": meta_extra["lineage_group"],
            "seed_ids": meta_extra.get("seed_ids", []),
            "license_ref": meta_extra.get("license_ref", "generated-cc0")}
    return {"id": rec_id, "messages": messages, "metadata": meta}


def gen_from_seed(seed, n):
    """n hybrid HTTP records from one observed seed; lineage_group = the seed id."""
    out = []
    for i in range(n):
        rec_id = f"{seed['id']}~{i}"
        tel, prov = wrap_http({**seed, "id": rec_id})
        out.append(assemble(rec_id, seed["class"], tel, prov,
                            {"method": "scenario_oracle", "oracle": "http_wrap",
                             "derived_from": seed.get("parent_id"), "derivation": seed.get("transforms", []),
                             "transforms": [t for t, rx in (("host_neutralize", SHORTCUT_HOST), ("entity_rename", CLASS_TOKEN))
                                            if rx.search(seed["payload"])],
                             "sink": "raw_block" if seed["class"] == "http_request_smuggling" else "per_record"},
                            {"source": seed["source"], "source_revision": seed["source_revision"],
                             "original_id": seed["id"], "license_ref": seed["license_ref"] or "unknown",
                             "lineage_group": seed.get("cluster", seed["id"]), "seed_ids": [seed["id"]],
                             "template_family": f"{seed['class']}_httpwrap"}))
    return out


def gen_flow(cls, n):
    out = []
    for i in range(n):
        rec_id = f"gen:flow:{cls}:{i:06d}"
        tel, scenario = synth_flow(cls, i)
        out.append(assemble(rec_id, cls, tel, "synthetic",
                            {"method": "scenario_oracle", "oracle": "flow_profile", "scenario": scenario},
                            {"lineage_group": rec_id, "template_family": f"{cls}_flow"}))
    return out


def gen_benign_std(cls, n):
    out = []
    for i in range(n):
        rec_id = f"gen:http:{cls}:{i:06d}"
        tel, scenario = synth_benign_http(cls, i)
        out.append(assemble(rec_id, cls, tel, "synthetic",
                            {"method": "scenario_oracle", "oracle": "benign_http", "scenario": scenario},
                            {"lineage_group": rec_id, "template_family": f"{cls}_http"}))
    return out


def gen_observed_http(entries):
    """CRS rule fixtures -> records. Label evidence = the rule author's expected rule id."""
    out = []
    for e in entries:
        rec_id = f"crs:{e['original_id']}"
        out.append(assemble(rec_id, e["cls"], e["telemetry"], "hybrid",
                            {"method": "source_annotation", "source_label": e["original_label"],
                             "note": "rule match is an annotation signal, not universal truth"},
                            {"source": e["source"], "source_revision": e["source_revision"],
                             "original_id": e["original_id"], "original_label": e["original_label"],
                             "license_ref": e["license_ref"], "lineage_group": e["lineage_group"],
                             "template_family": "crs_fixture"}))
    return out


def gen_observed_flow(entries):
    """Observed flow rows (data_pipeline.flows pool) -> records. Label from the source dataset."""
    out = []
    for e in entries:
        rec_id = f"obs:{e['source']}:{e['original_id']}".replace(" ", "_")[:128]
        out.append(assemble(rec_id, e["cls"], e["telemetry"], "observed",
                            {"method": "source_annotation", "source_label": e["original_label"]},
                            {"source": e["source"], "source_revision": e["source_revision"],
                             "original_id": e["original_id"], "original_label": e["original_label"],
                             "license_ref": e["license_ref"], "lineage_group": e["lineage_group"],
                             "template_family": f"{e['source']}_flow"}))
    return out


# ---- Deterministic SQL obfuscation (spec §6 role 1: documented transformations, parent recorded) ----
SQL_KW = re.compile(r"(?i)\b(select|union|from|where|and|or|order|by|sleep|null|insert|having|group|concat|benchmark)\b")


def _case_mix(m, r):
    return "".join(c.upper() if r.random() < 0.5 else c.lower() for c in m.group(0))


SQL_OBF = {
    "inline_comment": lambda t, r: SQL_KW.sub(lambda m: m.group(0) + "/**/", t).replace("/**/ ", "/**/"),
    "case_mix": lambda t, r: SQL_KW.sub(lambda m: _case_mix(m, r), t),
    "ws_substitute": lambda t, r: re.sub(r" +", lambda m: r.choice(["%09", "%0a", "%0d%0a", "+", "%a0"]), t),
    "url_encode": lambda t, r: __import__("urllib.parse").parse.quote(t, safe=""),
    "double_url_encode": lambda t, r: __import__("urllib.parse").parse.quote(
        __import__("urllib.parse").parse.quote(t, safe=""), safe=""),
}


def derive_obfuscated_sqli(classic_seeds):
    """sqli_classic seed -> obfuscated variant (1-2 transforms, must change the string).
    Lineage stays with the parent's near-dup cluster, so parent and child share one split."""
    out = []
    for sd in classic_seeds:
        r = rng_for("obf:" + sd["id"])
        names = r.sample(["inline_comment", "case_mix", "ws_substitute"], r.choice([1, 2]))
        if r.random() < 0.35:
            names.append(r.choice(["url_encode", "double_url_encode"]))  # encoding last
        t = sd["payload"]
        for n in names:
            t = SQL_OBF[n](t, r)
        if t == sd["payload"] or not SQL_KW.search(sd["payload"]):
            continue  # no obfuscation applied / no SQL keyword to obfuscate -> not a valid child
        out.append({**sd, "id": f"derived:sqli_obfuscated:{hashlib.sha256(t.encode()).hexdigest()[:16]}",
                    "class": "sqli_obfuscated", "payload": t, "parent_id": sd["id"], "transforms": names})
    return out


def load_seeds(cls):
    p = SEEDS / f"{cls}.jsonl"
    seeds = [json.loads(l) for l in open(p)] if p.exists() else []
    from subagents.review import status
    rv = status()
    seeds = [sd for sd in seeds if rv.get(sd["id"]) != "quarantined"]  # blind-review rejects
    if cls == "sqli_obfuscated":  # 11 observed seeds survive filtering; add deterministic derivatives
        seeds += derive_obfuscated_sqli(load_seeds("sqli_classic"))
    return seeds

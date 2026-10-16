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
import math
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
SHORTCUT_HOST = re.compile(r"[A-Za-z0-9.-]*(?:evil|attacker|malicious|hacker|vulnerable|victim|exploit)"
                           r"[A-Za-z0-9-]*(?:\.[A-Za-z]{2,})+", re.I)
NEUTRAL_HOSTS = ["cdn-static.net", "files.partner-site.org", "img.example.org", "assets.shopfront.io",
                 "status.vendor-portal.com", "media.newsdesk.net"]


# Class-name tokens that appear in seeds by convention only (XXE entity names, smuggling probe headers).
CLASS_TOKEN = re.compile(r"\bxxe\b|smuggl\w*", re.I)
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
_RES = ["products", "orders", "items", "users", "accounts", "invoices", "reports", "files", "comments",
        "articles", "tickets", "projects", "customers", "carts", "messages", "events", "assets", "teams"]
ENDPOINTS = sorted({"/search", "/login", "/logout", "/upload", "/render", "/redirect", "/download", "/feedback",
                    "/export", "/import", "/preview", "/callback", "/oauth/authorize", "/sso/return", "/checkout",
                    "/account/settings", "/user/profile", "/docs/view", "/admin/report", "/admin/users",
                    "/help/article", "/blog/post", "/contact", "/newsletter/subscribe", "/cart/add",
                    "/index.php", "/view.php", "/cgi-bin/status", "/wp-admin/admin-ajax.php", "/servlet/report",
                    "/api/search", "/api/v1/search", "/api/v2/search", "/rest/v1/query", "/internal/health"}
                   | {f"/api/{v}/{res}" for v in ("v1", "v2") for res in _RES}
                   | {f"/{res}" for res in _RES} | {f"/{res}/view" for res in _RES[:10]})
HOSTS = ["app.example.test", "shop.example.test", "api.example.test", "portal.example.test", "www.example.test",
         "m.example.test", "admin.example.test", "store.example.test", "docs.example.test", "help.example.test",
         "billing.example.test", "auth.example.test", "cdn.example.test", "intranet.corp.test", "crm.corp.test",
         "erp.corp.test", "git.corp.test", "wiki.corp.test", "api.partner.test", "status.partner.test"]
PARAMS = ["q", "id", "name", "next", "file", "page", "sort", "lang", "template", "url", "ref", "filter", "user",
          "path", "query", "redirect_to", "search", "keyword", "term", "category", "cat", "item", "pid", "uid",
          "order", "dir", "view", "type", "format", "callback", "return", "returnUrl", "dest", "target", "doc",
          "title", "comment", "message", "email", "username", "action", "mode", "theme", "locale", "region",
          "from", "to", "limit", "offset", "fields", "include", "expand", "token_hint", "session_hint"]
LOCALES = ["en-US,en;q=0.9", "en-GB,en;q=0.8", "de-DE,de;q=0.8", "ja-JP", "es-ES,es;q=0.9", "ar-SA", "fr-FR",
           "pt-BR", "zh-CN", "ko-KR", "ru-RU", "it-IT", "nl-NL", "tr-TR", "hi-IN", "pl-PL", "sv-SE", "fa-IR"]
# User-agent families x versions: ~20 families, hundreds of strings; shared by every class.
_BROWSER_ACCEPT = ["Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"]
UA_FAMILIES = {
    "chrome_win": lambda r: f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{r.randint(118, 142)}.0.0.0 Safari/537.36",
    "chrome_mac": lambda r: f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{r.randint(118, 142)}.0.0.0 Safari/537.36",
    "chrome_android": lambda r: f"Mozilla/5.0 (Linux; Android {r.randint(11, 16)}; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{r.randint(118, 142)}.0.0.0 Mobile Safari/537.36",
    "edge": lambda r: f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{(v := r.randint(118, 142))}.0.0.0 Safari/537.36 Edg/{v}.0.0.0",
    "firefox_linux": lambda r: f"Mozilla/5.0 (X11; Linux x86_64; rv:{(v := r.randint(115, 142))}.0) Gecko/20100101 Firefox/{v}.0",
    "firefox_win": lambda r: f"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{(v := r.randint(115, 142))}.0) Gecko/20100101 Firefox/{v}.0",
    "safari_mac": lambda r: f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{r.randint(16, 19)}.{r.randint(0, 6)} Safari/605.1.15",
    "safari_ios": lambda r: f"Mozilla/5.0 (iPhone; CPU iPhone OS {(v := r.randint(16, 19))}_{r.randint(0, 6)} like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{v}.0 Mobile/15E148 Safari/604.1",
    "curl": lambda r: f"curl/{r.randint(7, 8)}.{r.randint(50, 88) if r.random() < 0.3 else r.randint(0, 12)}.{r.randint(0, 3)}",
    "wget": lambda r: f"Wget/1.{r.randint(20, 25)}.{r.randint(0, 3)}",
    "python_requests": lambda r: f"python-requests/2.{r.randint(25, 33)}.{r.randint(0, 4)}",
    "httpx": lambda r: f"python-httpx/0.{r.randint(24, 28)}.{r.randint(0, 2)}",
    "okhttp": lambda r: f"okhttp/{r.choice(['3.14.9', '4.9.3', '4.11.0', '4.12.0', '5.0.0'])}",
    "go": lambda r: r.choice(["Go-http-client/1.1", "Go-http-client/2.0"]),
    "axios": lambda r: f"axios/1.{r.randint(4, 9)}.{r.randint(0, 9)}",
    "node": lambda r: r.choice(["node-fetch/1.0 (+https://github.com/bitinn/node-fetch)", "undici", "node"]),
    "java": lambda r: f"Java-http-client/{r.choice(['11.0.22', '17.0.12', '21.0.4'])}",
    "apache_hc": lambda r: f"Apache-HttpClient/{r.choice(['4.5.14', '5.2.1', '5.3.1'])} (Java/{r.choice(['11.0.22', '17.0.12', '21.0.4'])})",
    "powershell": lambda r: f"Mozilla/5.0 (Windows NT; Windows NT 10.0; en-US) WindowsPowerShell/5.1.{r.randint(17763, 26100)}.1",
    "postman": lambda r: f"PostmanRuntime/7.{r.randint(32, 43)}.{r.randint(0, 3)}",
    "dart": lambda r: f"Dart/{r.randint(2, 3)}.{r.randint(0, 9)} (dart:io)",
}
_BROWSERS = {"chrome_win", "chrome_mac", "chrome_android", "edge", "firefox_linux", "firefox_win", "safari_mac", "safari_ios"}
# kept for callers that pick a named profile; values are sampled per record
HEADER_PROFILES = UA_FAMILIES
# allowed payload positions per class (where the mechanism can actually be delivered)
SINKS = {"sqli_classic": ["query", "form", "json", "cookie", "multipart", "referer", "path"],
         "sqli_obfuscated": ["query", "form", "json", "cookie", "multipart", "referer"],
         "nosqli": ["json", "query", "form"], "xss_reflected_stored": ["query", "form", "json", "multipart", "referer", "cookie"],
         "xss_dom_polyglot": ["query", "form", "json", "referer"], "rce_cmd_injection": ["query", "form", "json", "multipart", "cookie"],
         "ssti": ["query", "form", "json", "multipart"], "xxe": ["xml", "multipart_xml"], "crlf_header_injection": ["query", "path", "referer"],
         "ldap_xpath_injection": ["query", "form", "json"], "path_traversal_lfi": ["query", "path", "cookie", "multipart"],
         "open_redirect_cors": ["query", "form"]}
BENIGN_SINKS = ["query", "query", "form", "json", "cookie", "xml", "multipart", "referer", "path"]
# extra encoding layer an attacker may add on top of transport encoding (not for SQL: an encoded SQLi
# is sqli_obfuscated by taxonomy precedence). Applied to benign values at a lower rate so the layer
# itself is not a label.
EXTRA_ENCODE = {"xss_reflected_stored", "xss_dom_polyglot", "rce_cmd_injection", "ssti", "path_traversal_lfi",
                "crlf_header_injection", "ldap_xpath_injection", "nosqli", "open_redirect_cors"}
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
    fam = r.choice(sorted(UA_FAMILIES))
    hs = [["User-Agent", UA_FAMILIES[fam](r)]]
    if fam in _BROWSERS:
        hs += [list(_BROWSER_ACCEPT), ["Accept-Encoding", r.choice(["gzip, deflate, br", "gzip, deflate, br, zstd"])],
               ["Accept-Language", r.choice(LOCALES)]]
        if fam.startswith(("chrome", "edge")) and r.random() < 0.7:
            hs += [["Sec-Fetch-Mode", r.choice(["navigate", "cors", "same-origin"])],
                   ["Sec-Fetch-Site", r.choice(["same-origin", "same-site", "none", "cross-site"])]]
        if r.random() < 0.5:
            hs.append(["Connection", "keep-alive"])
    else:
        hs.append(["Accept", r.choice(["*/*", "application/json", "application/json, text/plain, */*"])])
        if r.random() < 0.6:
            hs.append(["Accept-Encoding", r.choice(["gzip", "gzip, deflate", "gzip, deflate, br"])])
        if r.random() < 0.3:
            hs.append(["Accept-Language", r.choice(LOCALES)])
    if r.random() < 0.25:
        hs.append(["X-Request-Id", "%08x-%04x-%04x-%04x-%012x" % tuple(r.getrandbits(b) for b in (32, 16, 16, 16, 48))])
    if r.random() < 0.15:
        hs.append(["X-Forwarded-For", f"{r.randint(1, 223)}.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"])
    r.shuffle(hs)  # header order is client/proxy dependent; Host stays first like most clients
    return [["Host", host], *hs]


# Neutral, class-agnostic deployment context: attached to any HTTP record at the same rate so the mere
# presence of `context` / `history` carries no label signal (shortcut audit: context_keys, has_history).
def neutral_context(r, tel):
    if r.random() < 0.45:
        ctx = tel.setdefault("context", {})
        ctx.setdefault("edge", {"tls_version": r.choice(["TLSv1.2", "TLSv1.3", "TLSv1.3"]),
                                "client_asn_type": r.choice(["residential", "mobile", "hosting", "business"]),
                                "geo": r.choice(["US", "DE", "JP", "BR", "IN", "FR", "GB", "NL", "SG", "ZA"])})
        if r.random() < 0.5:
            ctx.setdefault("route", {"auth": r.choice(["none", "session", "bearer", "api_key"]),
                                     "rate_limit_tier": r.choice(["default", "strict", "partner"])})
        if tel.get("http") and r.random() < 0.5:
            tel["mode"] = "HYBRID_CTX"  # mode tag follows the presence of context, not the class
    if r.random() < 0.3:
        tel.setdefault("history", {}).setdefault("client_requests_60s", r.choice([1, 2, 3, 5, 8, 13, 21, 40]))
    return tel


def _benign_pairs(r, k):
    keys = r.sample(sorted(BENIGN_VALUES), k)
    return [[key, r.choice(BENIGN_VALUES[key])] for key in keys]


def _json_unicode_escape(s):
    return "".join(f"\\u{ord(c):04x}" if c in "<>'\"&;`$(){}|/\\" else c for c in s)


def build_request(r, sink, param, value, extra, layer=None):
    """One HTTP request with `value` placed at `sink`. raw target = what the client sent;
    query/body pairs = the server's single-pass decoding; decoded_views only for deeper layers.
    layer: optional extra client-side encoding ("double_url" | "json_unicode"), recorded as a view."""
    from urllib.parse import quote, unquote, unquote_plus
    host, path = r.choice(HOSTS), r.choice(ENDPOINTS)
    headers = _headers(r, host)
    http = {"method": "GET", "target": path, "protocol": r.choice(["HTTP/1.1", "HTTP/1.1", "HTTP/2"]),
            "headers": headers, "query": None, "cookies": None, "body": None, "body_encoding": "none"}
    views, stage = [], "request_headers"
    orig = value
    if layer == "double_url" and sink not in ("json", "xml", "multipart_xml"):
        value = quote(value, safe="")  # the server's first decode yields this; a second decode yields orig
    enc = quote(value, safe=QUOTE_SAFE)
    pairs_raw = [[k, quote(v, safe=QUOTE_SAFE)] for k, v in extra]
    if sink == "referer":  # reflected-in-logs / analytics sinks read the Referer query
        headers.append(["Referer", f"https://{r.choice(HOSTS)}{r.choice(ENDPOINTS)}?{param}={enc}"])
        views.append({"field": "header.referer", "decoder": "url", "depth": 1, "value": value})
        if extra:
            http["target"] += "?" + "&".join(f"{k}={v}" for k, v in pairs_raw)
            http["query"] = [[k, unquote_plus(v)] for k, v in pairs_raw]
    elif sink in ("query", "path"):
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
            if r.random() < 0.4:  # nested objects / arrays are where real APIs carry fields
                obj[r.choice(["data", "filter", "payload", "input", "params"])] = (
                    {param: value} if r.random() < 0.7 else [{param: value}])
            else:
                obj[param] = value
            body = json.dumps(obj, ensure_ascii=False)
            if layer == "json_unicode":
                body = body.replace(json.dumps(value, ensure_ascii=False)[1:-1], _json_unicode_escape(value), 1)
                views.append({"field": f"body.{param}", "decoder": "json_unicode", "depth": 1, "value": value})
            http["body"] = body
            headers.append(["Content-Type", r.choice(["application/json", "application/json; charset=utf-8"])])
        elif sink in ("multipart", "multipart_xml"):
            b = "----WebKitFormBoundary" + "".join(r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789") for _ in range(16))
            parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n' for k, v in extra]
            if sink == "multipart_xml":
                fname = r.choice(["import.xml", "feed.xml", "settings.xml", "data.svg", "report.docx.xml"])
                parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\n'
                             f'Content-Type: {"image/svg+xml" if fname.endswith(".svg") else "application/xml"}\r\n\r\n{value}\r\n')
            else:
                parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{param}"\r\n\r\n{value}\r\n')
            r.shuffle(parts)
            http["body"] = "".join(parts) + f"--{b}--\r\n"
            headers.append(["Content-Type", f"multipart/form-data; boundary={b}"])
        elif sink == "xml":
            http["body"] = value
            headers.append(["Content-Type", r.choice(["application/xml", "text/xml; charset=utf-8"])])
        http["body_encoding"] = "utf8"
        if r.random() < 0.12:  # legitimate chunked uploads: Transfer-Encoding is not a smuggling-only header
            data = http["body"]
            http["body"] = f"{len(data.encode()):x}\r\n{data}\r\n0\r\n\r\n"
            headers.append(["Transfer-Encoding", "chunked"])
            views.append({"field": "body", "decoder": "chunked", "depth": 1, "value": data})
        else:
            headers.append(["Content-Length", str(len(http["body"].encode()))])
    # second decoding layer (double-encoded input): record it as a view, never replace the original
    for k, v in (http["query"] or []):
        if "%" in v and unquote(v) != v:
            views.append({"field": f"query.{k}", "decoder": "url", "depth": 2, "value": unquote(v)})
    if layer == "double_url" and sink in ("form", "cookie", "multipart", "referer", "path"):
        views.append({"field": f"{sink}.{param}", "decoder": "url", "depth": 2, "value": orig})
    if views:
        http["decoded_views"] = views
    tel = {"schema_version": 1, "mode": "L7_HTTP", "decision_stage": stage,
           "observed_until": "request_received", "truncated": False, "http": http}
    return neutral_context(r, tel)


def wrap_http_raw(seed):
    """Smuggling fixture: keep the framing headers (CL/TE, duplicates, obfuscation) and the body bytes
    verbatim, in order. The outer request line's path and Host come from the shared pools and the
    client's ordinary headers are interleaved, so layout/host/path carry no label signal. `body` is
    what follows the header block (it may contain a smuggled prefix). A fixture, not proof of a desync (§5)."""
    r = rng_for(seed["id"])
    raw, _ = neutralize_hosts(seed["payload"], r)
    head, _, body = raw.partition("\n\n")
    lines = head.split("\n")
    first = lines[0].split(" ")
    method = first[0] if first and first[0] else "POST"
    proto = first[2] if len(first) > 2 else "HTTP/1.1"
    host = r.choice(HOSTS)
    framing = []
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            if k.strip().lower() != "host":
                framing.append([k.strip(), v.strip()])
    client = [h for h in _headers(r, host)[1:] if h[0].lower() not in {k.lower() for k, _ in framing}]
    merged = list(framing)
    for h in client:  # interleave client headers without reordering the framing headers
        merged.insert(r.randint(0, len(merged)), h)
    http = {"method": method, "target": r.choice(ENDPOINTS), "protocol": proto, "headers": [["Host", host], *merged],
            "query": None, "cookies": None, "body": body, "body_encoding": "utf8"}
    tel = {"schema_version": 1, "mode": "L7_HTTP", "decision_stage": "request_headers_body",
           "observed_until": "request_received", "truncated": False, "http": http}
    return neutral_context(r, tel), "hybrid"


GENERIC_BENIGN = ["summer sale", "8812", "true", "2026-10-01", "blue", "John Smith", "monthly", "home",
                  "a7f3c9e1", "release notes", "Größe 42", "café crème", "#fff", "v2", "draft", "42",
                  "new arrivals", "support", "ok", "user@example.test", "1,2,3", "Q3 report"]
REDIRECT_BENIGN = ["/account", "/cart?step=2", "/orders/88", "https://app.example.test/home",
                   "/dashboard#billing", "https://shop.example.test/products/77"]
REDIRECT_PARAMS = {"next", "url", "redirect_to", "ref", "return", "returnUrl", "dest", "target", "callback"}


def benign_value(param, r):
    if param in REDIRECT_PARAMS:
        return r.choice(REDIRECT_BENIGN)
    return r.choice(BENIGN_VALUES.get(param) or GENERIC_BENIGN)


# Which store a route queries: required evidence for the query-injection classes ("database-backed input",
# "known document-store backend", directory/XML query). Attack records get a consistent backend on most
# records; benign records (incl. twins) and other classes get any backend, so its presence is not a label.
BACKENDS = {"sql": ["postgresql", "mysql", "mssql", "oracle", "sqlite"], "document": ["mongodb", "couchdb", "elasticsearch"],
            "directory": ["openldap", "active_directory"], "xml": ["xml_xpath_store", "saml_metadata_store"]}
CLASS_BACKEND = {"sqli_classic": ["sql"], "sqli_obfuscated": ["sql"], "nosqli": ["document"],
                 "ldap_xpath_injection": ["directory", "xml"]}


def add_backend(r, tel, cls=None):
    kinds = CLASS_BACKEND.get(cls)
    if kinds and r.random() < 0.7:
        kind = r.choice(kinds)
    elif r.random() < 0.3:
        kind = r.choice(sorted(BACKENDS))
    else:
        return tel
    tel.setdefault("context", {}).setdefault("route", {})["query_backend"] = r.choice(BACKENDS[kind])
    if tel.get("http"):
        tel["mode"] = "HYBRID_CTX"
    return tel


def wrap_http(seed, twin=False):
    """Seed -> request. twin=True: the SAME request (host, path, headers, sink, param, extra fields,
    encoding layer) carrying an ordinary value instead of the payload -> a minimal-pair benign record."""
    if seed["class"] == "http_request_smuggling":
        return wrap_http_raw(seed)
    r = rng_for(seed["id"])
    cls = seed["class"]
    p, _ = neutralize_hosts(seed["payload"], r)
    sink = r.choice(SINKS[cls])
    param = r.choice(sorted(REDIRECT_PARAMS)) if cls == "open_redirect_cors" else r.choice(PARAMS)
    extra = _benign_pairs(r, r.choice([0, 0, 1, 2]))
    extra = [kv for kv in extra if kv[0] != param]
    layer = None
    if cls in EXTRA_ENCODE and r.random() < 0.2:
        layer = "json_unicode" if sink == "json" else "double_url"
    if twin:
        tr = rng_for("twin:" + seed["id"])
        p = tr.choice(BENIGN_XML) if sink in ("xml", "multipart_xml") else benign_value(param, tr)
    tel = build_request(r, sink, param, p, extra, layer)
    return add_backend(rng_for("backend:" + seed["id"]), tel, cls), "hybrid"  # twin: same backend as its attack


# ---- Flow synthesis (L4 classes) -------------------------------------------------------------
# Each profile derives a consistent aggregate from a bounded scenario. Rates = counts/duration.
FLOW_PROFILES = {
    "syn_flood_ddos": dict(fwd_frac=(0.97, 1.0), ports=[80, 443], window="fixed_window", dur=(4, 6), pkts=(8000, 60000), proto="tcp",
                           syn_frac=(0.95, 1.0), bytes_per_pkt=(40, 60), dst_fanout=(1, 2),
                                    variants=[dict(name="low_rate_pulse", pkts=(1500, 6000), dur=(4, 6)), dict(name="syn_ack_mix", syn_frac=(0.7, 0.9)),
                                    dict(name="multi_target", dst_fanout=(3, 20), ports=[80, 443, 8080, 8443])]),
    "udp_amplification": dict(fwd_frac=(0.0, 0.05), ports=[53, 123, 1900, 11211], window="fixed_window", dur=(4, 6), pkts=(2000, 20000), proto="udp",
                              syn_frac=(0, 0), bytes_per_pkt=(800, 4000), dst_fanout=(1, 3),
                                    variants=[dict(name="cldap_chargen", ports=[389, 19]), dict(name="low_volume_reflection", pkts=(600, 2000)),
                                       dict(name="large_fragments", bytes_per_pkt=(1400, 4000))]),
    "slowloris_slowhttp": dict(fwd_frac=(0.6, 0.8), ports=[80, 443, 8080], window="fixed_window", dur=(20, 60), pkts=(50, 400), proto="tcp",
                               syn_frac=(0.05, 0.2), bytes_per_pkt=(60, 200), dst_fanout=(1, 1),
                                    variants=[dict(name="slow_post_body", fwd_frac=(0.75, 0.9), bytes_per_pkt=(40, 120)), dict(name="slow_read", fwd_frac=(0.3, 0.45), dur=(60, 300))]),
    "port_scan_recon": dict(fwd_frac=(0.5, 0.65), window="fixed_window", dur=(2, 8), pkts=(500, 5000), proto="tcp",
                            syn_frac=(0.8, 1.0), bytes_per_pkt=(40, 44), dst_fanout=(200, 2000),
                                    variants=[dict(name="udp_scan", proto="udp", syn_frac=(0, 0), bytes_per_pkt=(28, 80)), dict(name="slow_scan", dur=(60, 600), pkts=(100, 800), dst_fanout=(50, 300)),
                                              # moderate rate: fan-out, not packet rate, is the scan evidence (v7 test_unseen: slow scans read as benign)
                                              dict(name="paced_scan", dur=(15, 60), pkts=(300, 2000), dst_fanout=(100, 600))]),
    "credential_stuffing_bruteforce": dict(fwd_frac=(0.45, 0.6), ports=[21, 22, 443, 3389], window="fixed_window", dur=(10, 60), pkts=(200, 5000),
                                           proto="tcp", syn_frac=(0.1, 0.3), bytes_per_pkt=(300, 900),
                                           dst_fanout=(1, 2),
                                    variants=[dict(name="ssh_bruteforce", ports=[22], bytes_per_pkt=(150, 600)), dict(name="rdp_spray", ports=[3389], dur=(60, 600)),
                                                    dict(name="web_login_stuffing", ports=[443], pkts=(1000, 8000), bytes_per_pkt=(400, 1200), fwd_frac=(0.5, 0.7)),  # request-heavy: failed logins
                                                    dict(name="mail_login_spray", ports=[993, 995, 143, 110, 587], pkts=(300, 4000), bytes_per_pkt=(200, 700)),
                                                    dict(name="api_token_bruteforce", ports=[443, 8443], pkts=(500, 6000), bytes_per_pkt=(350, 900), fwd_frac=(0.5, 0.7)),
                                                    dict(name="vpn_portal_spray", ports=[443, 10443, 4433], pkts=(400, 5000), bytes_per_pkt=(450, 1300), dur=(30, 120), fwd_frac=(0.5, 0.7)),
                                                    dict(name="ftp_bruteforce", ports=[21], pkts=(200, 2500), bytes_per_pkt=(60, 200))]),
    "c2_beaconing_anomaly": dict(fwd_frac=(0.45, 0.6), ports=[443, 8443, 80], window="full_flow", dur=(300, 1800), pkts=(20, 120), proto="tcp",
                                 syn_frac=(0.02, 0.1), bytes_per_pkt=(200, 1200), dst_fanout=(1, 1),
                                 periodic=True,
                                    variants=[dict(name="long_poll_http", ports=[80, 8080], dur=(900, 3600), pkts=(60, 300)), dict(name="sparse_beacon", pkts=(8, 30), dur=(1200, 3600))]),
    "dns_tunneling_exfil": dict(fwd_frac=(0.45, 0.55), ports=[53], window="full_flow", dur=(30, 300), pkts=(200, 3000), proto="udp",
                                syn_frac=(0, 0), bytes_per_pkt=(120, 300), dst_fanout=(1, 1), dns=True,
                                    variants=[dict(name="txt_bulk", bytes_per_pkt=(300, 600)), dict(name="slow_drip", dur=(600, 3600), pkts=(30, 200)),
                                              dict(name="domain_rotation", dns_top=(0.08, 0.4))]),  # tunnel spread over several parent domains: top share is not the evidence
    "tls_ja4_ua_spoofing": dict(fwd_frac=(0.35, 0.55), ports=[443], window="full_flow", dur=(1, 30), pkts=(20, 400), proto="tcp",
                                syn_frac=(0.02, 0.08), bytes_per_pkt=(400, 1400), dst_fanout=(1, 1),
                                ja4=True,
                                    variants=[dict(name="short_handshake_burst", dur=(0.2, 3), pkts=(8, 40)), dict(name="long_session", dur=(60, 900), pkts=(200, 3000))]),
    # synthetic only (no observed source records HTTP/2 frames): stream floods over few established connections
    "http2_rapid_reset_continuation": dict(fwd_frac=(0.55, 0.85), ports=[443, 8443], window="fixed_window", dur=(4, 6), pkts=(5000, 80000), proto="tcp",
                                           syn_frac=(0.0, 0.01), bytes_per_pkt=(60, 400), dst_fanout=(1, 1), h2_rst=(0.85, 1.0), h2_cont=(0, 2), h2_rst_ms=(0.05, 3),
                                    variants=[dict(name="continuation_flood", pkts=(300, 6000), bytes_per_pkt=(900, 1460), fwd_frac=(0.85, 1.0), h2_rst=(0.0, 0.1), h2_cont=(40, 20000), h2_rst_ms=(20, 5000)),  # not a reset attack: ordinary cancel delays
                                              dict(name="mixed_reset", h2_rst=(0.45, 0.85)),  # overlaps legitimate stream cancellation
                                              dict(name="low_rate_reset", pkts=(800, 5000), bytes_per_pkt=(200, 900), h2_rst_ms=(0.1, 40)),  # paced resets: latency alone is not the rule
                                              dict(name="multi_connection", conn=(20, 200))]),
    "benign_standard_traffic": dict(fwd_frac=(0.25, 0.6), ports=[443, 80, 8080, 22, 993, 5432], window="full_flow", dur=(1, 120), pkts=(10, 500), proto="tcp",
                                    syn_frac=(0.01, 0.05), bytes_per_pkt=(400, 1400), dst_fanout=(1, 2),
                                    variants=[  # benign UDP/DNS/QUIC/NTP so protocol and port are not attack-only
                                        dict(name="dns_lookup", proto="udp", ports=[53], dur=(0.01, 2), pkts=(2, 12), bytes_per_pkt=(60, 300), fwd_frac=(0.4, 0.6), syn_frac=(0, 0), dns=True),
                                        # one app polling its own backend: most queries to one parent domain (observed benign captures mix a whole network)
                                        dict(name="dns_single_app", proto="udp", ports=[53], window="full_flow", dur=(30, 900), pkts=(20, 400), bytes_per_pkt=(70, 260), fwd_frac=(0.45, 0.55), syn_frac=(0, 0), dns=True, dns_top=(0.45, 0.98)),
                                        dict(name="quic_web", proto="udp", ports=[443], dur=(1, 90), pkts=(20, 900), bytes_per_pkt=(200, 1350), fwd_frac=(0.25, 0.5), syn_frac=(0, 0)),
                                        dict(name="ntp_sync", proto="udp", ports=[123], dur=(0.01, 1), pkts=(2, 4), bytes_per_pkt=(76, 90), fwd_frac=(0.5, 0.5), syn_frac=(0, 0)),
                                        dict(name="windowed_web", window="fixed_window", dur=(4, 6), pkts=(5, 300)),
                                        dict(name="dns_resolver_tcp", ports=[53], dur=(0.05, 3), pkts=(6, 40), bytes_per_pkt=(80, 900), dns=True),
                                        dict(name="doh_lookup", ports=[443], dur=(0.2, 20), pkts=(8, 60), bytes_per_pkt=(150, 700)),
                                        dict(name="ssdp_mdns_local", proto="udp", ports=[1900, 5353], dur=(0.5, 30), pkts=(2, 60), bytes_per_pkt=(120, 450), fwd_frac=(0.5, 0.9), syn_frac=(0, 0)),
                                        dict(name="smtp_session", ports=[25, 587, 465], dur=(1, 60), pkts=(15, 400), bytes_per_pkt=(100, 1400)),
                                        dict(name="db_session", ports=[5432, 3306, 1433, 27017], dur=(1, 600), pkts=(10, 3000), bytes_per_pkt=(80, 1400)),
                                        dict(name="health_check_poll", window="full_flow", dur=(300, 1800), pkts=(40, 200), bytes_per_pkt=(150, 600), fwd_frac=(0.45, 0.55), syn_frac=(0.05, 0.12), iat_cv=(0.16, 0.35)),  # periodic low-rate agent: ordinary traffic, not high-rate
                                        dict(name="metrics_push", ports=[443, 4317, 9090], window="full_flow", dur=(300, 3600), pkts=(60, 400), bytes_per_pkt=(300, 1200), fwd_frac=(0.55, 0.75), syn_frac=(0.02, 0.08), iat_cv=(0.16, 0.5)),  # periodic low-rate agent: ordinary traffic, not high-rate
                                        dict(name="keepalive_long_poll", ports=[443, 8443], window="full_flow", dur=(600, 3600), pkts=(30, 160), bytes_per_pkt=(150, 900), fwd_frac=(0.4, 0.6), syn_frac=(0.01, 0.06), iat_cv=(0.2, 0.6)),  # periodic low-rate agent: ordinary traffic, not high-rate
                                        dict(name="mobile_push_channel", ports=[443, 5223], window="full_flow", dur=(900, 3600), pkts=(20, 120), bytes_per_pkt=(120, 700), fwd_frac=(0.35, 0.55), syn_frac=(0.01, 0.05), iat_cv=(0.25, 0.8)),  # periodic low-rate agent: ordinary traffic, not high-rate
                                        dict(name="large_download_https", ports=[443, 80], dur=(10, 600), pkts=(1000, 40000), bytes_per_pkt=(1200, 1460), fwd_frac=(0.05, 0.2), syn_frac=(0.0, 0.01)),
                                        dict(name="h2_large_headers", ports=[443], dur=(1, 60), pkts=(10, 300), h2_rst=(0.0, 0.2), h2_cont=(2, 30)),  # big cookies / JWTs split over CONTINUATION
                                        dict(name="h2_browser_navigation", ports=[443], dur=(1, 120), pkts=(30, 800), h2_rst=(0.15, 0.6), h2_cont=(0, 2), h2_rst_ms=(30, 8000))]),  # cancelled prefetch / media seeks
    "benign_high_rate_legit": dict(fwd_frac=(0.3, 0.6), ports=[443, 80], window="fixed_window", dur=(4, 6), pkts=(2000, 15000), proto="tcp",
                                   syn_frac=(0.02, 0.08), bytes_per_pkt=(500, 1400), dst_fanout=(1, 3),
                                   variants=[  # hard negatives against floods, amplification and beacons
                                       dict(name="bulk_backup", ports=[22, 443, 873], window="full_flow", dur=(300, 3600), pkts=(50000, 900000), bytes_per_pkt=(1200, 1460), fwd_frac=(0.6, 0.7)),
                                       dict(name="dns_resolver_burst", proto="udp", ports=[53], pkts=(1500, 12000), bytes_per_pkt=(80, 400), fwd_frac=(0.45, 0.55), syn_frac=(0, 0), dns=True),
                                       # CDN / telemetry SDK burst: hashed, high-entropy labels under one parent domain
                                       dict(name="dns_cdn_burst", proto="udp", ports=[53], pkts=(1000, 8000), bytes_per_pkt=(90, 300), fwd_frac=(0.45, 0.55), syn_frac=(0, 0), dns=True, dns_top=(0.4, 0.95), dns_len=(16, 40), dns_ent=(3.2, 4.2)),
                                       dict(name="video_stream_udp", proto="udp", ports=[443, 3478], pkts=(3000, 20000), bytes_per_pkt=(900, 1350), fwd_frac=(0.05, 0.2), syn_frac=(0, 0)),
                                       # legitimate login and API bursts: the lookalike of credential stuffing
                                       dict(name="sso_login_peak", ports=[443], pkts=(800, 6000), bytes_per_pkt=(400, 1100), fwd_frac=(0.25, 0.42), syn_frac=(0.05, 0.15), conn=(2, 20)),  # responses dominate: successful logins load pages
                                       dict(name="api_batch_sync", ports=[443, 8443], pkts=(2000, 10000), bytes_per_pkt=(600, 1400), fwd_frac=(0.2, 0.4), syn_frac=(0.01, 0.05), conn=(2, 30)),
                                       dict(name="software_update_cdn", ports=[80, 443], window="full_flow", dur=(30, 900), pkts=(5000, 120000), bytes_per_pkt=(1200, 1460), fwd_frac=(0.05, 0.2), syn_frac=(0.0, 0.01)),
                                       # sibling download regimes: no benign behaviour rests on a single variant
                                       dict(name="backup_restore_download", ports=[443, 22], window="full_flow", dur=(300, 3600), pkts=(20000, 500000), bytes_per_pkt=(1200, 1460), fwd_frac=(0.05, 0.2), syn_frac=(0.0, 0.01)),
                                       dict(name="mirror_sync_pull", ports=[80, 443, 873], window="full_flow", dur=(600, 3600), pkts=(50000, 900000), bytes_per_pkt=(1300, 1460), fwd_frac=(0.04, 0.15), syn_frac=(0.0, 0.01)),
                                       dict(name="dataset_fetch_windowed", ports=[443], pkts=(3000, 15000), bytes_per_pkt=(1200, 1460), fwd_frac=(0.05, 0.2), syn_frac=(0.0, 0.02)),
                                       # legitimate high stream-cancel rates: the lookalike of rapid reset
                                       dict(name="grpc_stream_cancel", ports=[443, 8443], pkts=(2000, 30000), bytes_per_pkt=(120, 700), fwd_frac=(0.45, 0.7), syn_frac=(0.0, 0.01), h2_rst=(0.2, 0.7), h2_cont=(0, 2), conn=(1, 8), h2_rst_ms=(1, 3000)),
                                       dict(name="h2_api_gateway_fanin", ports=[443], pkts=(4000, 40000), bytes_per_pkt=(150, 900), fwd_frac=(0.4, 0.65), syn_frac=(0.0, 0.01), h2_rst=(0.0, 0.3), h2_cont=(0, 6), conn=(1, 6), h2_rst_ms=(0.3, 2000))]),  # hedged requests: losers cancelled at once
}
# Window evidence per class: (connections to the same dst host+port in the minute, distinct dst ports on that host).
# None for ports = follow the profile's dst_fanout. Variants may override with "conn" / "dports".
WINDOW_EVIDENCE = {
    "syn_flood_ddos": ((800, 60000), (1, 2)), "udp_amplification": ((1, 12), (1, 2)),
    "slowloris_slowhttp": ((60, 1200), (1, 1)), "port_scan_recon": ((1, 3), None),
    "credential_stuffing_bruteforce": ((40, 900), (1, 2)), "c2_beaconing_anomaly": ((1, 3), (1, 1)),
    "dns_tunneling_exfil": ((1, 6), (1, 1)), "tls_ja4_ua_spoofing": ((1, 12), (1, 1)),
    "http2_rapid_reset_continuation": ((1, 30), (1, 1)),
    "benign_standard_traffic": ((1, 12), (1, 3)), "benign_high_rate_legit": ((1, 40), (1, 3)),
}


def _log_uniform(r, lo, hi):
    import math
    return max(lo, min(hi, int(round(math.exp(r.uniform(math.log(lo), math.log(hi + 1)))))))


# TLS client evidence (taxonomy tls_ja4_ua_spoofing: measured fingerprint + justified mapping + UA as received).
# Fingerprints are JA4-format placeholders, one per client family (not real-world values).
def _ja4(fam):
    h = hashlib.sha256(fam.encode()).hexdigest()
    return f"t13d{15 + len(fam) % 4}{10 + len(fam) % 7}h2_{h[:12]}_{h[12:24]}"


TLS_BROWSERS = ["chrome", "firefox", "safari", "edge"]
TLS_TOOLS = ["python_requests", "go_http", "curl", "java_httpclient", "node_undici", "headless_automation_lib"]
TLS_PROXIES = ["corporate_tls_inspection_proxy", "mobile_carrier_optimizer"]


def tls_context(r, kind):
    """kind: spoof (tool stack claiming a browser UA), match (stack and UA agree), proxy (legitimate mismatch:
    a re-encrypting middlebox the deployment knows about) -> context block with the evidence."""
    ua_fam = r.choice(TLS_BROWSERS) if kind != "match" or r.random() < 0.7 else r.choice(TLS_TOOLS)
    if kind == "spoof":
        stack = r.choice(TLS_TOOLS)
    elif kind == "match":
        stack = ua_fam
    else:
        stack = r.choice(TLS_PROXIES)
    ua = UA_FAMILIES.get({"chrome": "chrome_win", "firefox": "firefox_linux", "safari": "safari_mac", "edge": "edge",
                          "python_requests": "python_requests", "go_http": "go", "curl": "curl", "java_httpclient": "java",
                          "node_undici": "node"}.get(ua_fam, "chrome_win"), UA_FAMILIES["chrome_win"])(r)
    ctx = {"tls": {"ja4": _ja4(stack), "ja4_client_family": stack, "user_agent_received": ua,
                   "ua_claimed_family": ua_fam if ua_fam in TLS_BROWSERS else "non_browser"}}
    if kind == "proxy":
        ctx["egress"] = {"known_tls_inspection": True, "device": stack}
    # client behaviour in the window: impersonating tools rotate UAs at automation pace; a proxy carries many
    # users (many UAs) but is a known device; ordinary clients keep one or two
    uas, rpm = {"spoof": ((3, 40), (30, 600)), "proxy": ((8, 120), (20, 400)), "match": ((1, 2), (1, 60))}[kind]
    ctx["client_window_10m"] = {"distinct_user_agents": _log_uniform(r, *uas), "requests_per_min": _log_uniform(r, *rpm)}
    return ctx


# Optional evidence fields appear in every class where the measurement would exist, so their presence
# is not a label: values carry the evidence (e.g. low iat_cv or high qname entropy), not the key.
# Overlapping on purpose (CIC-Bell-DNS-EXF-2021, observed: exfil qname length 25 / entropy 2.44 vs benign 18.5 / 2.50):
# entropy alone does not separate real exfil; tunnels range from short low-entropy labels to long encoded ones,
# and benign includes high-entropy names (CDN hashes, DKIM, telemetry). Volume and rate carry the evidence.
# (qname_len_mean, entropy, parent_domain_top_share). Real CIC-Bell-DNS-EXF-2021 60 s windows (p10-p90): qname length
# benign 17.5-20, exfil 24.9-26; top share benign 0.19-0.29, exfil 0.51-0.62; entropy does not separate them.
# Synthetic ranges bracket the real ones with a small overlap; wider benign ranges contradicted observed exfil records.
DNS_NORMAL, DNS_TUNNEL = ((8, 24), (2.0, 3.9), (0.05, 0.45)), ((23, 120), (2.2, 4.5), (0.42, 0.98))


H2_NORMAL = ((0.0, 0.15), (0, 3), (20, 5000))  # (RST_STREAM/HEADERS ratio, CONTINUATION frames, HEADERS->RST ms) for ordinary HTTP/2 traffic


def sig(x, n=6):
    """Round to n significant digits: keeps relative precision for low and high rates."""
    return float(f"{x:.{n}g}")


def synth_flow(cls, idx):
    r = rng_for(f"flow:{cls}:{idx}")
    pr = FLOW_PROFILES[cls]
    if pr.get("variants") and r.random() < 0.5:
        pr = {**pr, **r.choice(pr["variants"])}
    pr = {"name": "base", **pr}
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
        "pkt_len_min": min(bpp, max(40, bpp - spread)), "pkt_len_max": bpp + spread,  # floor never above the mean (tiny UDP probes)
        "tcp_flags_bitmask": flags if pr["proto"] == "tcp" else 0,
        "ip_proto": 6 if pr["proto"] == "tcp" else 17,
        "dst_port": r.choice(pr["ports"]) if pr.get("ports") else r.randint(1, 65535),
    }
    if pr.get("periodic"):
        feats["iat_cv"] = round(r.uniform(0.0, 0.15), 4)  # low jitter -> beacon-like
    elif pr.get("iat_cv"):
        feats["iat_cv"] = round(r.uniform(*pr["iat_cv"]), 4)  # scheduled benign agents: periodic, jitterier than a beacon
    elif pr["window"] == "full_flow" and r.random() < 0.4:
        feats["iat_cv"] = round(r.uniform(0.4, 2.5), 4)
    if pr.get("dns") or (feats["dst_port"] == 53 and pr["proto"] == "udp"):
        (ln, ent, top) = DNS_TUNNEL if cls == "dns_tunneling_exfil" else DNS_NORMAL
        ln, ent, top = pr.get("dns_len", ln), pr.get("dns_ent", ent), pr.get("dns_top", top)
        feats["qname_len_mean"] = round(r.uniform(*ln), 2)
        feats["qname_entropy_bits_per_char"] = round(r.uniform(*ent), 3)
        feats["parent_domain_top_share"] = round(r.uniform(*top), 3)
    if pr.get("ja4"):
        feats["ja4_ua_consistent"] = 0.0  # measured mismatch; mapping justified in metadata
    elif pr["proto"] == "tcp" and feats["dst_port"] in (443, 8443) and r.random() < 0.4:
        feats["ja4_ua_consistent"] = 1.0
    # HTTP/2 frame counters: on any TLS flow that carried application data (not SYN floods or scans), so the
    # keys are not a label; the h2 class and the benign lookalike variants set their own ranges
    if "h2_rst" in pr or (pr["proto"] == "tcp" and feats["dst_port"] in (443, 8443)
                          and cls not in ("syn_flood_ddos", "port_scan_recon") and r.random() < 0.5):
        lo, hi = pr.get("h2_cont", H2_NORMAL[1])
        feats["h2_rst_stream_ratio"] = round(r.uniform(*pr.get("h2_rst", H2_NORMAL[0])), 3)
        feats["h2_continuation_frames"] = _log_uniform(r, lo, hi) if lo else r.randint(lo, hi)
        if feats["h2_rst_stream_ratio"] > 0:  # rapid reset cancels in the same write as HEADERS; real cancels wait on something
            lo, hi = pr.get("h2_rst_ms", H2_NORMAL[2])
            feats["h2_rst_latency_ms_p50"] = round(math.exp(r.uniform(math.log(lo), math.log(hi))), 3)
    conn_rng, port_rng = WINDOW_EVIDENCE[cls]
    conn_rng = pr.get("conn", conn_rng)
    port_rng = pr.get("dports", port_rng or pr.get("dst_fanout", (1, 1)))
    feats["conn_count_60s"] = _log_uniform(r, *conn_rng)
    feats["distinct_dst_ports_60s"] = _log_uniform(r, *port_rng)
    tel = {"schema_version": 1, "mode": "L4_FLOW",
           "decision_stage": "flow_complete" if pr["window"] == "full_flow" else "flow_window",
           "observed_until": pr["window"], "truncated": False,
           "flow": {"window": {"kind": pr["window"], "seconds": None if pr["window"] == "full_flow" else dur},
                    "protocol": pr["proto"], "features": feats}}
    if "ja4_ua_consistent" in feats:  # the flag never stands alone: the measured evidence travels with it
        kind = "spoof" if cls == "tls_ja4_ua_spoofing" else "match"
        if cls == "benign_standard_traffic" and r.random() < 0.3:  # legitimate mismatch: mismatch alone is not malicious
            kind = "proxy"
            feats["ja4_ua_consistent"] = 0.0
        tel["context"] = tls_context(r, kind)
        tel["mode"] = "HYBRID_CTX"
    scenario = {"profile": cls, "variant": pr["name"], "duration_s": dur, "pkts": pkts, "bytes_per_pkt": bpp,
                "syn_fraction_range": pr["syn_frac"], "window_kind": pr["window"]}
    return tel, scenario


# ---- Benign HTTP (same endpoints / headers / params as attacks) ---------------------------------
def synth_benign_http(cls, idx):
    r = rng_for(f"benign:{cls}:{idx}")
    pairs = _benign_pairs(r, r.choice([1, 1, 2, 3]))
    param, value = pairs[-1]
    if r.random() < 0.5:  # widen the parameter vocabulary beyond the BENIGN_VALUES keys
        param = r.choice(PARAMS)
        value = benign_value(param, r)
        pairs = [kv for kv in pairs[:-1] if kv[0] != param] + [[param, value]]
    sink = r.choice(BENIGN_SINKS)
    if sink == "xml" or (sink == "multipart" and r.random() < 0.3):  # legit XML: content type is not an XXE signal
        value, sink = r.choice(BENIGN_XML), ("xml" if sink == "xml" else "multipart_xml")
    layer = None
    if r.random() < 0.08:
        layer = "json_unicode" if sink == "json" else "double_url"
    tel = add_backend(r, build_request(r, sink, param, value, pairs[:-1], layer))
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
    from data_pipeline.mechanisms import tags
    meta["mechanism_tags"] = tags(tel, evidence, meta)
    return {"id": rec_id, "messages": messages, "metadata": meta}


TWIN_RATE = 0.3  # share of seeded attack records that get a benign minimal-pair twin


def gen_twin(seed, rec_id):
    """Benign minimal pair: identical request minus the payload. Same lineage group as the attack
    record, so the pair never straddles a split."""
    tel, _ = wrap_http({**seed, "id": rec_id}, twin=True)
    return assemble(f"twin:{rec_id}", "benign_standard_traffic", tel, "synthetic",
                    {"method": "scenario_oracle", "oracle": "minimal_pair_twin", "twin_of": rec_id},
                    {"lineage_group": seed.get("cluster", seed["id"]), "template_family": "minimal_pair_twin"})


def gen_from_seed(seed, n):
    """n hybrid HTTP records from one observed seed; lineage_group = the seed id."""
    out = []
    for i in range(n):
        rec_id = f"{seed['id']}~{i}"
        if seed["class"] != "http_request_smuggling" and rng_for("twin?" + rec_id).random() < TWIN_RATE:
            out.append(gen_twin(seed, rec_id))
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
        # schema id charset is [A-Za-z0-9_.:~-]; source ids may carry '@', '/', spaces (CIC-DDoS2019 flow id + timestamp)
        rec_id = re.sub(r"[^A-Za-z0-9_.:~-]", "_", f"obs:{e['source']}:{e['original_id']}")[:128]
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
    from subagents.review import relabels, status
    rv, rel = status(), relabels()
    seeds = [sd for sd in seeds if rv.get(sd["id"]) != "quarantined"  # blind-review rejects
             and rel.get(sd["id"]) != "__quarantine__"]  # manual rejects (relabel.json)
    if cls == "sqli_obfuscated":  # 11 observed seeds survive filtering; add deterministic derivatives
        seeds += derive_obfuscated_sqli(load_seeds("sqli_classic"))
    return seeds

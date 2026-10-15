"""Coverage tags (spec §7 metadata `mechanism_tags`): where and how a record delivers its content.
Tags drive the coverage matrix (eval.coverage); they never enter model input.

Tags are structural, taken from what the generator already knows rather than from payload text:
  loc:<sink>        delivery location (query, path, form, json, multipart, cookie, referer, xml, raw_block, flow)
  enc:<decoder>     extra decoding layers recorded in decoded_views (url depth 2, json_unicode, chunked)
  src:<source>      seed source list / dataset (seeded and observed records)
  family:<name>     scenario family declared by a Haiku worker (scenario.family)
  variant:<name>    synthetic flow profile variant
"""
import re


def _loc(tel):
    h = tel.get("http")
    if not h:
        return "flow" if tel.get("flow") else "none"
    hd = {k.lower(): v for k, v in h["headers"]}
    ct = hd.get("content-type", "").split(";")[0]
    if h.get("body"):
        if "multipart" in ct:
            return "multipart"
        if "json" in ct:
            return "json"
        if "xml" in ct:
            return "xml"
        if "form" in ct:
            return "form"
        return "raw_body"
    if any(v["field"] == "header.referer" for v in h.get("decoded_views", [])):
        return "referer"
    if h.get("cookies"):
        return "cookie"
    if any(v["field"] == "target.path" for v in h.get("decoded_views", [])):
        return "path"
    return "query" if h.get("query") else "request_line"


def tags(tel, evidence, meta):
    out = {f"loc:{_loc(tel)}"}
    for v in (tel.get("http") or {}).get("decoded_views", []):
        if v["decoder"] != "url" or v["depth"] > 1:
            out.add(f"enc:{v['decoder']}{v['depth'] if v['decoder'] == 'url' else ''}")
    if meta.get("source") not in (None, "generated"):
        out.add("src:" + re.sub(r"[^a-z0-9_.-]+", "_", str(meta["source"]).lower())[:48])
    scen = (evidence or {}).get("scenario") or {}
    if scen.get("family"):
        out.add("family:" + re.sub(r"[^a-z0-9_]+", "_", str(scen["family"]).lower())[:48])
    if scen.get("variant"):
        out.add(f"variant:{scen['variant']}")
    for t in (evidence or {}).get("derivation") or []:
        out.add(f"transform:{t}")
    return sorted(out)


if __name__ == "__main__":
    tel = {"http": {"headers": [["Content-Type", "application/json"]], "body": "{}",
                    "decoded_views": [{"field": "body.q", "decoder": "json_unicode", "depth": 1, "value": "x"}]}}
    assert tags(tel, {"scenario": {"family": "Nested Batch"}}, {"source": "generated"}) == \
        ["enc:json_unicode", "family:nested_batch", "loc:json"]
    assert tags({"flow": {"features": {}}}, {"scenario": {"variant": "dns_lookup"}}, {}) == ["loc:flow", "variant:dns_lookup"]
    print("ok")

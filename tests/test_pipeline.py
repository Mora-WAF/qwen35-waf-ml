import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_pipeline import generate as G  # noqa: E402
from data_pipeline.dedup_and_split import assign_splits, dedup, lineage_overlap  # noqa: E402
from data_pipeline.normalize import parse_http_blocks, parse_lines  # noqa: E402


def test_parse_lines_skips_comments_and_blanks():
    assert list(parse_lines("# hdr\n\n a \n---\nb\n// c\n")) == ["a", "b"]


def test_http_blocks_keep_framing_and_inner_blank_lines():
    raw = "POST / HTTP/1.1\nA: 1\nA: 2\n\n0\n\nX\n\n\nGET / HTTP/1.1\nB: 3\n"
    blocks = list(parse_http_blocks(raw))
    assert len(blocks) == 2 and blocks[0] == "POST / HTTP/1.1\nA: 1\nA: 2\n\n0\n\nX"


def test_raw_smuggling_wrap_preserves_duplicates_and_body():
    raw = "POST /x HTTP/1.1\nTransfer-Encoding: chunked\nTransfer-Encoding: identity\n\n0\n\nZ"
    tel, prov = G.wrap_http_raw({"id": "t", "payload": raw, "class": "http_request_smuggling"})
    assert tel["http"]["headers"] == [["Transfer-Encoding", "chunked"], ["Transfer-Encoding", "identity"]]
    assert tel["http"]["body"] == raw and prov == "hybrid"


import yaml  # noqa: E402
VOCAB = set(yaml.safe_load(open(Path(__file__).parent.parent / "config/features.yaml"))["features"])


def test_flow_features_consistent():
    for cls in G.FLOW_PROFILES:
        for i in range(25):
            f = G.synth_flow(cls, i)[0]["flow"]["features"]
            assert all(math.isfinite(v) and v >= 0 for v in f.values()), cls
            assert f["duration_s"] > 0
            assert math.isclose(f["pkts_per_s"], f["pkts_total"] / f["duration_s"], rel_tol=1e-3)
            assert math.isclose(f["bytes_per_s"], f["bytes_total"] / f["duration_s"], rel_tol=1e-3)
            assert 0 <= f["tcp_flags_bitmask"] <= 255
            assert f["pkts_fwd"] + f["pkts_bwd"] == f["pkts_total"] and f["pkt_len_min"] <= f["pkt_len_max"]
            assert set(f) <= VOCAB, set(f) - VOCAB


def _rec(cls, grp, content):
    return {"messages": [{}, {"content": content}, {}],
            "metadata": {"threat_type": cls, "lineage_group": grp}}


def test_groups_never_straddle_splits_and_dups_dropped():
    recs = [_rec(c, f"{c}:{g}", f"{c}{g}{k}") for c in ("a", "b") for g in range(10) for k in range(3)]
    recs.append(_rec("a", "a:0", "a00"))  # exact dup
    kept, dropped = dedup(recs)
    assert dropped == 1
    out = assign_splits(kept)
    assert not lineage_overlap(out)
    for c in ("a", "b"):
        got = {r["metadata"]["split"] for r in out if r["metadata"]["threat_type"] == c}
        assert got == {"train", "test", "val_select", "val_calib", "val_policy"}, got


def test_shortcut_hosts_neutralized():
    import random
    t, changed = G.neutralize_hosts("//evil.com/x https://app.attacker.example ok.org", random.Random(0))
    assert changed and "evil" not in t and "attacker" not in t and "ok.org" in t
    seed = {"id": "s1", "class": "open_redirect_cors", "payload": "https://evil.com"}
    tel, _ = G.wrap_http(seed)
    assert "evil" not in str(tel)


def test_observed_flow_mappers():
    from data_pipeline.flows import cic_features, nf_features, FLAG_BITS
    base = {"Flow Duration": 2_000_000, "Flow IAT Min": 5, "Flow IAT Mean": 10, "Flow IAT Max": 30,
            "Total Fwd Packets": 3, "Total Backward Packets": 1, "Total Length of Fwd Packets": 100,
            "Total Length of Bwd Packets": 0, "Packet Length Mean": 25, "Min Packet Length": 0,
            "Max Packet Length": 60, "Init_Win_bytes_forward": -1, "destination_port": 22,
            "protocol": "tcp", **{c: 0 for c in FLAG_BITS}, "SYN Flag Count": 1, "ACK Flag Count": 1}
    f, why = cic_features(base)
    assert why is None and f["duration_s"] == 2.0 and f["pkts_per_s"] == 2.0  # us -> s, recomputed rate
    assert f["tcp_flags_bitmask"] == 2 | 16                                   # presence flags -> bitmask
    assert set(f) <= VOCAB
    assert cic_features({**base, "Flow Duration": -1})[1] == "negative_or_nonfinite_duration"
    assert cic_features({**base, "Flow IAT Min": -3})[1] == "negative_iat"
    nf = {"FLOW_DURATION_MILLISECONDS": 0, "IN_PKTS": 2, "OUT_PKTS": 0, "IN_BYTES": 80, "OUT_BYTES": 0,
          "SHORTEST_FLOW_PKT": 40, "LONGEST_FLOW_PKT": 40, "TCP_FLAGS": 2, "L4_DST_PORT": 80, "PROTOCOL": 6,
          "MIN_TTL": 64, "MAX_TTL": 64, "RETRANSMITTED_IN_PKTS": 0, "RETRANSMITTED_OUT_PKTS": 0}
    f, why = nf_features(nf)
    assert why is None and f["duration_s"] == 0 and "pkts_per_s" not in f   # zero duration -> no rate
    assert set(f) <= VOCAB


def test_group_spanning_classes_stays_in_one_split():
    recs = [_rec(c, f"{c}:{g}", f"{c}{g}") for c in ("a", "b") for g in range(10)]
    recs += [_rec("a", "shared", "sa"), _rec("b", "shared", "sb")]  # one attacker, two classes
    out = assign_splits(dedup(recs)[0])
    assert not lineage_overlap(out)


def test_obfuscated_sqli_derivation():
    seeds = [{"id": f"s{i}", "class": "sqli_classic", "payload": p, "cluster": f"c{i}"}
             for i, p in enumerate(["' OR 1=1 -- ", "1 UNION SELECT null,version()", "admin'#"])]
    out = G.derive_obfuscated_sqli(seeds)
    assert out and all(o["payload"] != seeds[int(o["parent_id"][1:])]["payload"] for o in out)
    assert all(o["class"] == "sqli_obfuscated" and o["cluster"] == f"c{o['parent_id'][1:]}" for o in out)
    assert "s2" not in {o["parent_id"] for o in out}  # no SQL keyword -> no derivative


def test_request_builder_round_trip():
    import random
    from urllib.parse import unquote_plus
    r = random.Random(0)
    for sink in ("query", "form", "json", "cookie", "path"):
        tel = G.build_request(r, sink, "q", "a b'<x>%0d%0a", [["page", "2"]])
        h = tel["http"]
        if sink == "query":
            assert ["q", unquote_plus("a%20b'%3Cx%3E%0d%0a")] in h["query"] and "?" in h["target"]
            assert any(v["depth"] == 2 for v in h.get("decoded_views", [])) is False  # single layer only
        if sink in ("form", "json"):
            assert h["method"] in ("POST", "PUT") and tel["decision_stage"] == "request_headers_body"
            assert ["Content-Length", str(len(h["body"].encode()))] in h["headers"]


def test_observed_and_synthetic_share_core_names():
    core = {"duration_s", "pkts_total", "bytes_total", "pkts_fwd", "pkts_bwd", "bytes_fwd", "bytes_bwd",
            "pkts_per_s", "bytes_per_s", "pkt_len_min", "pkt_len_max", "tcp_flags_bitmask", "dst_port", "ip_proto"}
    extras = {"iat_cv", "qname_len_mean", "qname_entropy_bits_per_char", "ja4_ua_consistent"}
    for cls in G.FLOW_PROFILES:
        assert set(G.synth_flow(cls, 0)[0]["flow"]["features"]) - core <= extras, cls


def test_crs_fixture_to_telemetry():
    import json
    import jsonschema
    from data_pipeline.crs import telemetry
    inp = {"method": "POST", "uri": "/get?id=1&x=2", "version": "HTTP/1.1",
           "headers": {"Host": "localhost", "User-Agent": "OWASP CRS test agent", "Content-Type": "text/plain"},
           "data": "a=b"}
    tel = telemetry(inp, 942100, 1)
    jsonschema.validate(tel, json.load(open(Path(__file__).parent.parent / "schemas/telemetry.schema.json")))
    h = {k.lower(): v for k, v in tel["http"]["headers"]}
    assert h["host"] != "localhost" and "OWASP" not in h["user-agent"]   # test-harness headers normalized
    assert h["content-type"] == "text/plain" and tel["http"]["query"] == [["id", "1"], ["x", "2"]]
    assert tel["decision_stage"] == "request_headers_body"

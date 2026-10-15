import json
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
    hs = tel["http"]["headers"]
    assert [h for h in hs if h[0] == "Transfer-Encoding"] == [["Transfer-Encoding", "chunked"], ["Transfer-Encoding", "identity"]]
    assert hs[0][0] == "Host" and hs[0][1] in G.HOSTS and tel["http"]["target"] in G.ENDPOINTS  # pooled, not seed layout
    assert tel["http"]["body"] == "0\n\nZ" and prov == "hybrid"  # body = bytes after the header block, verbatim


def test_minimal_pair_twin_differs_only_in_value():
    seed = {"id": "s9", "class": "xss_reflected_stored", "payload": "<b>probe</b>"}
    a, _ = G.wrap_http(seed)
    b, _ = G.wrap_http(seed, twin=True)
    ha, hb = a["http"], b["http"]
    assert (ha["method"], ha["protocol"]) == (hb["method"], hb["protocol"])
    assert ha["headers"][0] == hb["headers"][0] and [k for k, _ in ha["headers"]] == [k for k, _ in hb["headers"]]
    assert "probe" in json.dumps(a) and "probe" not in json.dumps(b)


def test_haiku_style_fields_canonicalized():
    from subagents.orchestrator import canonicalize
    tel = {"schema_version": 1, "mode": "L7_HTTP", "decision_stage": "request_headers",
           "observed_until": "2026-10-14T08:00:00Z", "http": {"method": "GET", "target": "/", "protocol": "HTTP/2.0", "headers": []}}
    out = canonicalize(tel, "k")
    assert out["observed_until"] == "request_received" and out["http"]["protocol"] == "HTTP/2"


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
            assert (["Content-Length", str(len(h["body"].encode()))] in h["headers"]
                    or (["Transfer-Encoding", "chunked"] in h["headers"]
                        and any(v["decoder"] == "chunked" for v in h["decoded_views"])))


def test_observed_and_synthetic_share_core_names():
    core = {"duration_s", "pkts_total", "bytes_total", "pkts_fwd", "pkts_bwd", "bytes_fwd", "bytes_bwd",
            "pkts_per_s", "bytes_per_s", "pkt_len_min", "pkt_len_max", "tcp_flags_bitmask", "dst_port", "ip_proto"}
    extras = {"iat_cv", "qname_len_mean", "qname_entropy_bits_per_char", "parent_domain_top_share", "ja4_ua_consistent",
              "h2_rst_stream_ratio", "h2_continuation_frames",
              "conn_count_60s", "distinct_dst_ports_60s"}  # window aggregates: CIC has them too (from timestamps)
    for cls in G.FLOW_PROFILES:
        assert set(G.synth_flow(cls, 0)[0]["flow"]["features"]) - core <= extras, cls


def test_cic_window_counts():
    import pandas as pd
    from data_pipeline.flows import add_window_counts
    df = pd.DataFrame({"flow_id": range(5), "source_ip": ["a"] * 4 + ["b"], "destination_ip": ["h"] * 5,
                       "destination_port": [22, 22, 21, 22, 22],
                       "Timestamp": ["03/07/2017 08:55:58", "3/7/2017 8:55", "03/07/2017 08:55:01",
                                     "03/07/2017 08:57:00", "bad"]})
    out = add_window_counts(df)
    assert out["conn_count_60s"].tolist()[:4] == [2, 2, 1, 1]          # same src/dst/port in the same minute
    assert out["distinct_dst_ports_60s"].tolist()[:3] == [2, 2, 2]      # ports 21 and 22 in that minute
    assert out["conn_count_60s"].isna().tolist()[4]                     # unparsable timestamp -> no count


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


def test_review_status_unanswered_is_not_agreed(tmp_path, monkeypatch):
    from subagents import review as R
    (tmp_path / "out").mkdir()
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"r1": {"ref": "a:0"}, "r2": {"ref": "a:1"}, "r3": {"ref": "a:2"},
                               "r4": {"ref": "a:1"}, "r5": {"ref": "a:0"}}))  # r4, r5: recheck rounds
    (tmp_path / "out" / "batch_x.jsonl").write_text('{"rid": "r1"}\n{"rid": "r2"}\n{"rid": "r4"}\n')
    (tmp_path / "quarantine.json").write_text(json.dumps([{"rid": "r2", "ref": "a:1"}]))
    monkeypatch.setattr(R, "RV", tmp_path)
    monkeypatch.setattr(R, "KEY", key)
    # a:1 quarantined in round 1, cleared by the answered recheck r4; a:0's recheck r5 is unanswered -> r1 stands
    assert R.status() == {"a:0": "agreed", "a:1": "agreed", "a:2": "not_reviewed"}


def test_unseen_family_holdout_takes_whole_groups():
    from data_pipeline.dedup_and_split import holdout_unseen
    def rec(cls, fam, grp, gen="haiku-5.5"):
        return {"metadata": {"threat_type": cls, "template_family": f"{cls}:{fam}", "generator": gen,
                             "lineage_group": grp, "mechanism_tags": [], "provenance": "synthetic"}}
    recs = [rec("graphql_abuse", f"f{i}", f"g{i}") for i in range(10)]
    recs.append(rec("benign_standard_traffic", "f0_benign", "g0"))  # twin shares its attack's group
    out = holdout_unseen(recs, frac=0.2)
    assert len(out) == 2 and out <= {f"g{i}" for i in range(10)}
    assert holdout_unseen(recs[:2]) == set()  # fewer than 3 families: nothing held out


def test_ids2018_row_maps_through_cic_features():
    import pandas as pd
    from data_pipeline.flows import FLAG_BITS, IDS2018_RENAME, NUM_PROTO, cic_features
    row = {"Dst Port": 80, "Protocol": 6, "Flow Duration": 2_000_000, "Tot Fwd Pkts": 3, "Tot Bwd Pkts": 1,
           "TotLen Fwd Pkts": 300, "TotLen Bwd Pkts": 0, "Pkt Len Min": 0, "Pkt Len Max": 150, "Flow IAT Mean": 5.0,
           "Flow IAT Min": 1.0, "Label": "DoS attacks-Slowloris", **{k.replace("Count", "Cnt"): 0 for k in FLAG_BITS}}
    row["SYN Flag Cnt"] = 1
    r = pd.Series(row).rename(IDS2018_RENAME)
    r["protocol"] = NUM_PROTO[int(r["protocol"])]
    f, why = cic_features(r)
    assert why is None and f["duration_s"] == 2.0 and f["pkts_total"] == 4 and f["dst_port"] == 80
    assert f["tcp_flags_bitmask"] == 2 and f["ip_proto"] == 6 and "conn_count_60s" not in f  # no IPs -> no window count


def test_obfuscated_sql_seed_routing():
    from data_pipeline.normalize import is_obfuscated_sql
    for t in ("\\x27UNION SELECT", "'OR%0b1=1--", "x 0x730065006c00 exec(@q)", "UnIoN SeLeCt 1", "UN/**/ION SEL/**/ECT",
              "' or char(65)=char(65)"):
        assert is_obfuscated_sql(t), t
    for t in ('" OR "1"="1', "' or 1=1 --", "' UNION SELECT null--", "admin' -- ", "1 Union Select 1"):
        assert not is_obfuscated_sql(t), t
    from data_pipeline.normalize import sqli_seed_plain
    assert not is_obfuscated_sql(sqli_seed_plain("'))%20waitfor%20delay%20'0:0:20'%20--"))  # transport only
    assert is_obfuscated_sql(sqli_seed_plain("or%25201=1"))  # nested: a layer survives one decode


def test_flow_label_consistency():
    from data_pipeline.flows import CONSISTENT
    assert not CONSISTENT["syn_flood_ddos"]({"ip_proto": 6, "tcp_flags_bitmask": 16})  # ACK-only CIC-DDoS2019 "Syn"
    assert CONSISTENT["syn_flood_ddos"]({"ip_proto": 6, "tcp_flags_bitmask": 2})
    assert not CONSISTENT["udp_amplification"]({"ip_proto": 6})
    assert not CONSISTENT["udp_amplification"]({"ip_proto": 17, "pkts_fwd": 2, "pkts_bwd": 0})  # one-way spray
    assert not CONSISTENT["slowloris_slowhttp"]({"ip_proto": 6, "dst_port": 21})


def test_calibration_recovers_temperature_and_respects_hn_constraint():
    import numpy as np
    from eval.calibration import choose_tau, fit_temperature, softmax
    rng = np.random.default_rng(0)
    n, k, T_true = 4000, 6, 3.0
    logits = rng.normal(0, 2, (n, k))
    y = np.array([rng.choice(k, p=softmax(l)) for l in logits])
    assert abs(fit_temperature(logits * T_true, y) - T_true) < 0.25  # overconfident scores -> T near 3
    benign = np.array([True, True, False, False, False, False])
    hard_neg = np.array([False, True, False, False, False, False])
    tau, r = choose_tau(softmax(logits), y, benign, hard_neg, 0.01)
    assert r["hard_negative_false_intervention"] <= 0.01 and 0 <= tau <= 1


def test_make_args_builds_on_pinned_transformers(monkeypatch, tmp_path):
    # the notebook's TrainingArguments must construct on the pinned transformers (5.x renamed group_by_length)
    import inspect
    import transformers
    from training import colab_helpers as H
    real = set(inspect.signature(transformers.TrainingArguments.__init__).parameters)
    monkeypatch.setattr(H, "dtype_is_bf16", lambda: True)
    monkeypatch.setattr(transformers, "TrainingArguments", lambda **kw: kw)  # no GPU here: check names, not devices
    for b in (1, 2, 4):
        kw = H.make_args(str(tmp_path), 25, 10, smoke=True, per_device=b)
        assert set(kw) <= real, set(kw) - real
        assert kw["per_device_train_batch_size"] * kw["gradient_accumulation_steps"] == H.EFFECTIVE_BATCH

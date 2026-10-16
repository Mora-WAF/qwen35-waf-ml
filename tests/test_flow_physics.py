import copy
import json

from data_pipeline import flow_physics as P


def record(features, cls="benign_standard_traffic", provenance="synthetic", mode="L4_FLOW", rid="flow-0001"):
    tel = {"mode": mode, "flow": {"features": features}}
    return {"id": rid, "metadata": {"threat_type": cls, "provenance": provenance},
            "messages": [{"role": "user", "content": "[L4_FLOW] " + json.dumps(tel)}]}


def test_valid_missing_and_observed_mean():
    f = {"duration_s": 2, "pkts_total": 10, "pkts_fwd": 4, "pkts_bwd": 6,
         "bytes_total": 1000, "bytes_fwd": 400, "bytes_bwd": 600,
         "pkts_per_s": 5, "bytes_per_s": 500, "pkt_len_min": 50, "pkt_len_max": 150}
    assert P.check(f, "benign_standard_traffic", "synthetic") == ([], [])
    assert P.check({}, "syn_flood_ddos", "synthetic") == ([], [])
    f["bytes_total"] = 10000
    f.pop("bytes_fwd"); f.pop("bytes_per_s")
    assert P.check(f, "benign_standard_traffic", "synthetic")[0] == ["pkt_len_mean"]
    assert P.check(f, "benign_standard_traffic", "observed") == ([], [])


def test_arithmetic_zero_duration_and_tolerance():
    f = {"pkts_total": 100, "pkts_fwd": 40, "pkts_bwd": 59, "duration_s": 2,
         "pkts_per_s": 51, "bytes_total": 1000, "bytes_fwd": 400, "bytes_bwd": 580,
         "bytes_per_s": 510}
    assert set(P.check(f, "x", "synthetic")[0]) == {"bytes_sum", "bytes_per_s"}
    f.update(duration_s=0, pkts_per_s=0, bytes_per_s=0)
    assert set(P.check(f, "x", "synthetic")[0]) == {"bytes_sum", "pkts_per_s_zero_duration", "bytes_per_s_zero_duration"}


def test_ranges_protocol_and_warning_severity():
    f = {"ip_proto": 17, "tcp_flags_bitmask": 2, "dst_port": 65536, "pkts_total": 0,
         "h2_rst_stream_ratio": 1.01, "ja4_ua_consistent": 0.5, "conn_count_60s": 0,
         "h2_continuation_frames": -1, "pkts_fwd": 4, "pkts_bwd": 2}
    errors, warnings = P.check(f, "udp_amplification", "synthetic")
    assert set(errors) == {"pkts_sum", "udp_tcp_flags", "range:dst_port", "range:pkts_total",
                           "range:h2_rst_stream_ratio", "range:ja4_ua_consistent",
                           "range:conn_count_60s", "range:h2_continuation_frames"}
    assert warnings == ["udp_amplification_direction"]
    assert P.check({"ip_proto": 6, "tcp_flags_bitmask": 256}, "syn_flood_ddos", "synthetic") == (
        ["range:tcp_flags_bitmask"], ["class_consistency"])
    assert P.check({"ip_proto": 6}, "syn_flood_ddos", "synthetic") == ([], [])
    assert P.check({"ip_proto": 6, "tcp_flags_bitmask": 2.0}, "syn_flood_ddos", "synthetic") == ([], [])
    assert P.check({"ip_proto": 17}, "udp_amplification", "synthetic") == ([], [])


def test_report_counts_examples_modes_and_no_mutation(tmp_path, monkeypatch):
    records = [record({"pkt_len_min": 90, "pkt_len_max": 40}, rid=f"flow-{i:04}") for i in range(7)]
    records += [record({}, provenance="observed", mode="HYBRID_CTX"), record({}, mode="L7_HTTP")]
    before = copy.deepcopy(records)
    report = P.summarize(records)
    group = report["benign_standard_traffic"]
    assert group["synthetic"]["n"] == 7 and group["observed"]["n"] == 1
    assert group["synthetic"]["errors"]["pkt_len_order"] == {
        "n": 7, "ids": [f"flow-{i:04}" for i in range(5)]}
    assert records == before
    monkeypatch.setattr(P, "ROOT", tmp_path)
    (tmp_path / "datasets_ready").mkdir()
    source = tmp_path / "datasets_ready/fast_test.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    original = source.read_bytes()
    assert P.main("fast_") == 0
    assert json.loads((tmp_path / "reports/flow_physics_fast_.json").read_text()) == report
    assert source.read_bytes() == original

"""Report-only flow consistency: python -m data_pipeline.flow_physics [PREFIX].

Arithmetic tolerates 1% or one count/byte/rate unit, whichever is larger; ranges are strict.
Observed packet lengths and byte counts use unreconciled measurement layers. Hybrid
provenance carries generated context and is grouped with synthetic telemetry.
"""
import json
import math
import sys
from pathlib import Path

import yaml

from data_pipeline.flows import CONSISTENT

ROOT = Path(__file__).resolve().parent.parent
FEATURES = yaml.safe_load((ROOT / "config/features.yaml").read_text())["features"]
CLASS_INPUTS = {
    "syn_flood_ddos": ("ip_proto", "tcp_flags_bitmask"),
    "udp_amplification": ("ip_proto", "pkts_bwd"),
    "slowloris_slowhttp": ("ip_proto", "dst_port"),
    "credential_stuffing_bruteforce": ("ip_proto",),
    "port_scan_recon": ("ip_proto",),
    "dns_tunneling_exfil": ("dst_port",),
    "http_flood_app_dos": ("ip_proto", "dst_port", "conn_count_60s"),
}


def check(features, cls, provenance):
    errors, warnings = [], []
    f = {}
    for k, v in features.items():
        if k not in FEATURES:
            continue
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            errors.append(f"numeric:{k}")
            continue
        lo, hi = FEATURES[k].get("range", (None, None))
        if ((lo is not None and v < lo) or (hi is not None and v > hi)
                or (FEATURES[k]["type"] == "int" and v != int(v))):
            errors.append(f"range:{k}")
        if FEATURES[k]["type"] == "int":
            if v != int(v):
                continue
            v = int(v)  # CONSISTENT uses bit operations; JSON may encode integral values as floats.
        f[k] = v
    for stem in ("pkts", "bytes"):
        total, fwd, bwd, rate = f"{stem}_total", f"{stem}_fwd", f"{stem}_bwd", f"{stem}_per_s"
        if all(k in f for k in (total, fwd, bwd)):
            expected = f[fwd] + f[bwd]
            if not math.isclose(f[total], expected, rel_tol=0.01, abs_tol=1):
                errors.append(f"{stem}_sum")
        if "duration_s" in f:
            if f["duration_s"] == 0 and rate in features:
                errors.append(f"{rate}_zero_duration")
            elif f["duration_s"] > 0 and total in f and rate in f:
                if not math.isclose(f[rate], f[total] / f["duration_s"], rel_tol=0.01, abs_tol=1):
                    errors.append(rate)
    if "pkt_len_min" in f and "pkt_len_max" in f:
        lo, hi = f["pkt_len_min"], f["pkt_len_max"]
        if lo > hi:
            errors.append("pkt_len_order")
        if provenance != "observed" and "bytes_total" in f and f.get("pkts_total", 0) > 0:
            mean = f["bytes_total"] / f["pkts_total"]
            if mean < lo - max(1, abs(lo) * 0.01) or mean > hi + max(1, abs(hi) * 0.01):
                errors.append("pkt_len_mean")
    if "h2_rst_latency_ms_p50" in f and not f.get("h2_rst_stream_ratio"):
        errors.append("h2_rst_latency_without_rst")
    if f.get("ip_proto") == 17 and "tcp_flags_bitmask" in f and f["tcp_flags_bitmask"] != 0:
        errors.append("udp_tcp_flags")
    if cls in CONSISTENT and all(k in f for k in CLASS_INPUTS[cls]):
        if not CONSISTENT[cls](f):
            warnings.append("class_consistency")
    if cls == "udp_amplification" and all(k in f for k in ("pkts_fwd", "pkts_bwd")):
        if f["pkts_bwd"] <= f["pkts_fwd"]:
            warnings.append("udp_amplification_direction")
    return errors, warnings


def summarize(records):
    report = {}
    for r in records:
        content = next(m["content"] for m in r["messages"] if m["role"] == "user")
        tel = json.loads(content.split("] ", 1)[1])
        if tel["mode"] not in ("L4_FLOW", "HYBRID_CTX"):
            continue
        meta = r["metadata"]
        cls, provenance = meta["threat_type"], meta["provenance"]
        groups = report.setdefault(cls, {p: {"n": 0, "errors": {}, "warnings": {}}
                                         for p in ("observed", "synthetic")})
        group = groups["observed" if provenance == "observed" else "synthetic"]
        group["n"] += 1
        errors, warnings = check(tel.get("flow", {}).get("features", {}), cls, provenance)
        for severity, rules in (("errors", errors), ("warnings", warnings)):
            for rule in rules:
                entry = group[severity].setdefault(rule, {"n": 0, "ids": []})
                entry["n"] += 1
                if len(entry["ids"]) < 5:
                    entry["ids"].append(r["id"])
    return report


def main(prefix="fast_"):
    def records():
        for path in sorted((ROOT / "datasets_ready").glob(f"{prefix}*.jsonl")):
            with path.open() as fh:
                for line in fh:
                    yield json.loads(line)
    report = summarize(records())
    out = ROOT / "reports" / f"flow_physics_{prefix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "fast_"))

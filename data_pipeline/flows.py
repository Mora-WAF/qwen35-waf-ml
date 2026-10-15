"""Observed flow ingestion. Emits only the CORE vocabulary both sources share (config/features.yaml),
so feature names cannot identify the source. IAT/TTL/init-window columns are still read for
row validation but not emitted.

Observed flow ingestion: CIC-IDS2017 + NF-ToN-IoT-v2 -> shared-vocabulary telemetry (spec §3, §5).

- Only source labels that name a taxonomy class are mapped; everything else is counted as unmapped.
- Lineage = attack campaign / host: CIC (src_ip, dst_ip) — every CIC attack class is ONE campaign,
  so it lands wholly in one split; NF-ToN groups by attacker src_ip (23 scanning, 7 password hosts).
- Rows with negative durations/IATs are rejected, sentinels dropped, rates recomputed from duration.

usage: python -m data_pipeline.flows [--cap 3000]   -> data/flows/observed_pool.jsonl + reports/flow_ingest.json
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT.parent / "raw_cache"
OUT = ROOT / "data" / "flows"

CIC_MAP = {"BENIGN": "benign_standard_traffic", "PortScan": "port_scan_recon",
           "DoS slowloris": "slowloris_slowhttp", "DoS Slowhttptest": "slowloris_slowhttp",
           "FTP-Patator": "credential_stuffing_bruteforce", "SSH-Patator": "credential_stuffing_bruteforce"}
NF_MAP = {"Benign": "benign_standard_traffic", "scanning": "port_scan_recon",
          "password": "credential_stuffing_bruteforce"}
FLAG_BITS = {"FIN Flag Count": 1, "SYN Flag Count": 2, "RST Flag Count": 4, "PSH Flag Count": 8,
             "ACK Flag Count": 16, "URG Flag Count": 32}  # CIC values are 0/1 presence flags
PROTO = {"tcp": 6, "udp": 17}
CIC_COLS = ["flow_id", "source_ip", "destination_ip", "destination_port", "protocol", "Flow Duration",
            "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
            "Total Length of Bwd Packets", "Packet Length Mean", "Min Packet Length", "Max Packet Length",
            "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min", "Init_Win_bytes_forward", *FLAG_BITS, "attack_label",
            "Timestamp"]
NF_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL", "IN_BYTES", "IN_PKTS", "OUT_BYTES",
           "OUT_PKTS", "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "MIN_TTL", "MAX_TTL",
           "SHORTEST_FLOW_PKT", "LONGEST_FLOW_PKT", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_PKTS", "Attack"]


def sig(x, n=6):
    return float(f"{x:.{n}g}")


def _rates(f):
    if f["duration_s"] > 0:
        f["pkts_per_s"] = sig(f["pkts_total"] / f["duration_s"])
        f["bytes_per_s"] = sig(f["bytes_total"] / f["duration_s"])
    return f


def add_window_counts(df):
    """Window evidence a NIDS keeps per source: connections to the same destination host and port, and distinct
    ports touched on that host, in the same clock minute. CIC timestamps mix second- and minute-resolution
    formats, so the window is the minute bucket. Rows without a parsable timestamp get no counts (absent keys)."""
    import pandas as pd
    minute = pd.to_datetime(df["Timestamp"], format="mixed", dayfirst=True, errors="coerce").dt.floor("min")
    df = df.assign(_min=minute)
    ok = df["_min"].notna()
    keys = ["source_ip", "destination_ip", "_min"]
    df.loc[ok, "conn_count_60s"] = df[ok].groupby(keys + ["destination_port"])["flow_id"].transform("size")
    df.loc[ok, "distinct_dst_ports_60s"] = df[ok].groupby(keys)["destination_port"].transform("nunique")
    return df.drop(columns="_min")


def cic_features(r):
    """One CIC row -> (features, reject_reason). Duration/IAT in microseconds."""
    dur, iat_min = r["Flow Duration"], r["Flow IAT Min"]
    if not (math.isfinite(dur) and dur >= 0):
        return None, "negative_or_nonfinite_duration"
    if iat_min < 0 or r["Flow IAT Mean"] < 0:
        return None, "negative_iat"
    pf, pb = int(r["Total Fwd Packets"]), int(r["Total Backward Packets"])
    bf, bb = int(r["Total Length of Fwd Packets"]), int(r["Total Length of Bwd Packets"])
    if min(pf, pb, bf, bb) < 0 or pf + pb == 0:
        return None, "invalid_counts"
    f = {"duration_s": sig(dur / 1e6), "pkts_total": pf + pb, "bytes_total": bf + bb,
         "pkts_fwd": pf, "pkts_bwd": pb, "bytes_fwd": bf, "bytes_bwd": bb,
         "pkt_len_min": int(r["Min Packet Length"]), "pkt_len_max": int(r["Max Packet Length"]),
         "tcp_flags_bitmask": sum(b for c, b in FLAG_BITS.items() if r[c] == 1),
         "dst_port": int(r["destination_port"])}
    if r["protocol"] in PROTO:
        f["ip_proto"] = PROTO[r["protocol"]]
    for k in ("conn_count_60s", "distinct_dst_ports_60s"):
        if k in r and r[k] == r[k]:  # NaN-safe: absent when the timestamp did not parse
            f[k] = int(r[k])
    return _rates(f), None


def nf_features(r):
    """One NF-ToN-IoT-v2 row -> (features, reject_reason). Duration in milliseconds; 0 is common."""
    dur = r["FLOW_DURATION_MILLISECONDS"]
    if dur < 0:
        return None, "negative_duration"
    pi, po, bi, bo = (int(r[c]) for c in ("IN_PKTS", "OUT_PKTS", "IN_BYTES", "OUT_BYTES"))
    if min(pi, po, bi, bo) < 0 or pi + po == 0:
        return None, "invalid_counts"
    f = {"duration_s": sig(dur / 1e3), "pkts_total": pi + po, "bytes_total": bi + bo,
         "pkts_fwd": pi, "pkts_bwd": po, "bytes_fwd": bi, "bytes_bwd": bo,
         "pkt_len_min": int(r["SHORTEST_FLOW_PKT"]), "pkt_len_max": int(r["LONGEST_FLOW_PKT"]),
         "tcp_flags_bitmask": int(r["TCP_FLAGS"]), "dst_port": int(r["L4_DST_PORT"]),
         "ip_proto": int(r["PROTOCOL"])}
    return _rates(f), None


def telemetry(features):
    return {"schema_version": 1, "mode": "L4_FLOW", "decision_stage": "flow_complete",
            "observed_until": "full_flow", "truncated": False,
            "flow": {"window": {"kind": "full_flow", "seconds": None}, "features": features}}


def round_robin(df, group_col, cap, seed):
    """Deterministic sample spreading picks across groups before repeating any group."""
    df = df.sample(frac=1, random_state=seed)
    df = df.assign(_rank=df.groupby(group_col).cumcount())
    return df.sort_values(["_rank"], kind="stable").head(cap)


def ingest(cap=3000, seed=1337):
    srcs = {s["id"]: s for s in yaml.safe_load(open(ROOT / "config/sources.yaml"))["sources"]}
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    pool = []

    # --- CIC-IDS2017 (parquet, ~2.8M rows; only needed columns loaded)
    s = srcs["cic_ids2017_mirror"]
    p = next((RAW / s["id"] / s["revision"]).rglob("CICIDS_Flow.parquet"))
    df = pq.read_table(p, columns=CIC_COLS).to_pandas()
    df = add_window_counts(df)  # over ALL flows (incl. unmapped classes): what a sensor would have counted
    rep = {"rows": len(df), "unmapped": {}, "classes": {}}
    rep["unmapped"] = df.loc[~df.attack_label.isin(CIC_MAP), "attack_label"].value_counts().to_dict()
    df = df[df.attack_label.isin(CIC_MAP)].copy()
    df["cls"] = df.attack_label.map(CIC_MAP)
    df["grp"] = "cic:" + df.source_ip + "->" + df.destination_ip
    for cls, d in df.groupby("cls"):
        picked, rej = [], Counter()
        for _, r in round_robin(d, "grp", cap * 2, seed).iterrows():  # 2x: room for rejects
            f, why = cic_features(r)
            if why:
                rej[why] += 1
                continue
            picked.append({"cls": cls, "telemetry": telemetry(f), "source": s["id"],
                           "source_revision": s["revision"], "license_ref": s["license_spdx"],
                           "original_id": str(r["flow_id"]), "original_label": r["attack_label"],
                           "lineage_group": r["grp"]})
            if len(picked) >= cap:
                break
        pool += picked
        rep["classes"][cls] = {"available": len(d), "accepted": len(picked), "rejected": dict(rej),
                               "groups": d.grp.nunique()}
    report["cic_ids2017_mirror"] = rep

    # --- NF-ToN-IoT-v2 (2 GB CSV, chunked; per-chunk sample keeps memory bounded)
    s = srcs["nf_ton_iot_v2_mirror"]
    p = next((RAW / s["id"] / s["revision"]).rglob("NF-ToN-IoT-v2-train.csv"))
    rep = {"rows": 0, "unmapped": Counter(), "classes": {}}
    keep = []
    for i, ch in enumerate(pd.read_csv(p, usecols=NF_COLS, chunksize=500_000)):
        rep["rows"] += len(ch)
        rep["unmapped"].update(ch.loc[~ch.Attack.isin(NF_MAP), "Attack"].value_counts().to_dict())
        ch = ch[ch.Attack.isin(NF_MAP)].copy()
        ch["row"] = ch.index
        ch["grp"] = "nf:" + ch.IPV4_SRC_ADDR
        for _, d in ch.groupby("Attack"):
            keep.append(round_robin(d, "grp", cap, seed + i))
    nf = pd.concat(keep)
    nf["cls"] = nf.Attack.map(NF_MAP)
    for cls, d in nf.groupby("cls"):
        picked, rej = [], Counter()
        for _, r in round_robin(d, "grp", cap * 2, seed).iterrows():
            f, why = nf_features(r)
            if why:
                rej[why] += 1
                continue
            picked.append({"cls": cls, "telemetry": telemetry(f), "source": s["id"],
                           "source_revision": s["revision"], "license_ref": s["license_spdx"],
                           "original_id": f"train.csv:{r['row']}", "original_label": r["Attack"],
                           "lineage_group": r["grp"]})
            if len(picked) >= cap:
                break
        pool += picked
        rep["classes"][cls] = {"sampled_candidates": len(d), "accepted": len(picked),
                               "rejected": dict(rej), "groups": d.grp.nunique()}
    rep["unmapped"] = dict(rep["unmapped"])
    report["nf_ton_iot_v2_mirror"] = rep

    with open(OUT / "observed_pool.jsonl", "w") as fh:
        for x in pool:
            fh.write(json.dumps(x) + "\n")
    report["zero_duration_note"] = "NF durations have ms resolution; duration_s == 0 -> rate features absent"
    (ROOT / "reports/flow_ingest.json").write_text(json.dumps(report, indent=1, default=int))
    return report


def load_pool():
    p = OUT / "observed_pool.jsonl"
    by_cls = defaultdict(list)
    if p.exists():
        for line in open(p):
            x = json.loads(line)
            by_cls[x["cls"]].append(x)
    return by_cls


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3000)
    rep = ingest(ap.parse_args().cap)
    for src in ("cic_ids2017_mirror", "nf_ton_iot_v2_mirror"):
        print(src, json.dumps({c: (v["accepted"], v["groups"], v["rejected"]) for c, v in rep[src]["classes"].items()}))

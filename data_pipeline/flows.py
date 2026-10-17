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
import os
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
    """One NetFlow row (NF v2, or v1 without packet-length columns) -> (features, reject_reason). Duration in
    milliseconds; 0 is common. v1 has no SHORTEST/LONGEST_FLOW_PKT: those features are left out, not invented."""
    dur = r["FLOW_DURATION_MILLISECONDS"]
    if dur < 0:
        return None, "negative_duration"
    pi, po, bi, bo = (int(r[c]) for c in ("IN_PKTS", "OUT_PKTS", "IN_BYTES", "OUT_BYTES"))
    if min(pi, po, bi, bo) < 0 or pi + po == 0:
        return None, "invalid_counts"
    f = {"duration_s": sig(dur / 1e3), "pkts_total": pi + po, "bytes_total": bi + bo,
         "pkts_fwd": pi, "pkts_bwd": po, "bytes_fwd": bi, "bytes_bwd": bo,
         "tcp_flags_bitmask": int(r["TCP_FLAGS"]), "dst_port": int(r["L4_DST_PORT"]),
         "ip_proto": int(r["PROTOCOL"])}
    if "SHORTEST_FLOW_PKT" in r:
        f |= {"pkt_len_min": int(r["SHORTEST_FLOW_PKT"]), "pkt_len_max": int(r["LONGEST_FLOW_PKT"])}
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


# CIC-IDS2018 / CIC-DDoS2019: map to the CIC-2017 column schema so cic_features() (validated) applies unchanged
IDS2018_MAP = {"Benign": "benign_standard_traffic", "DoS attacks-Slowloris": "slowloris_slowhttp",
               "DoS attacks-SlowHTTPTest": "slowloris_slowhttp"}
DDOS2019_MAP = {"BENIGN": "benign_standard_traffic", "Syn": "syn_flood_ddos",
                **{f"DrDoS_{k}": "udp_amplification" for k in ("DNS", "NTP", "SSDP", "LDAP", "SNMP", "NetBIOS", "MSSQL")}}
# A source label is kept only when the measured telemetry can show it (label-consistency; the Codex+Gemini final
# re-check marked these flows port scans). Measured on the sampled pool, 2026-10-15:
#   CIC-DDoS2019 "Syn": SYN Flag Count is 0 on every row (ACK only, random dst ports) -> no SYN evidence;
#   CIC-DDoS2019 DrDoS_*: 677 of 3,000 sampled flows are TCP, zero-byte (time-window labelling, not reflection); the
#     UDP rest is one-way (no response packets, source ports 564/634/900, not 53/123/389): no request-response
#     asymmetry to see, and both external reviewers read them as port scans. Dropped by user decision 2026-10-15;
#     udp_amplification is synthetic-only until a source shows both directions;
#   CIC-IDS2018 "DoS attacks-SlowHTTPTest": all dst port 21, zero payload (FTP traffic, not slow HTTP).
WEB_PORTS = {80, 443, 8000, 8080, 8443}
CONSISTENT = {
    "syn_flood_ddos": lambda f: f.get("ip_proto") == 6 and bool((f.get("tcp_flags_bitmask") or 0) & 2),
    "udp_amplification": lambda f: f.get("ip_proto") == 17 and (f.get("pkts_bwd") or 0) > 0,  # both directions seen
    "slowloris_slowhttp": lambda f: f.get("ip_proto") == 6 and f.get("dst_port") in WEB_PORTS,
    "credential_stuffing_bruteforce": lambda f: f.get("ip_proto") == 6,
    "port_scan_recon": lambda f: f.get("ip_proto") in (6, 17),
    "dns_tunneling_exfil": lambda f: f.get("dst_port") == 53,
}
IDS2018_RENAME = {"Dst Port": "destination_port", "Protocol": "protocol", "Tot Fwd Pkts": "Total Fwd Packets",
                  "Tot Bwd Pkts": "Total Backward Packets", "TotLen Fwd Pkts": "Total Length of Fwd Packets",
                  "TotLen Bwd Pkts": "Total Length of Bwd Packets", "Pkt Len Min": "Min Packet Length",
                  "Pkt Len Max": "Max Packet Length", "Pkt Len Mean": "Packet Length Mean",
                  "Init Fwd Win Byts": "Init_Win_bytes_forward", "Label": "attack_label",
                  **{k.replace("Count", "Cnt"): k for k in FLAG_BITS}}
NUM_PROTO = {6: "tcp", 17: "udp"}


def _pick(df, src, cls_map, cap, seed, rep):
    """Round-robin over lineage groups per class, QC via cic_features; returns pool entries."""
    out = []
    for cls, d in df.groupby("cls"):
        picked, rej = [], Counter()
        for _, r in round_robin(d, "grp", cap * 2, seed).iterrows():
            f, why = cic_features(r)
            if why:
                rej[why] += 1
                continue
            picked.append({"cls": cls, "telemetry": telemetry(f), "source": src["id"], "source_revision": src["revision"],
                           "license_ref": src["license_spdx"], "original_id": str(r["flow_id"]),
                           "original_label": r["attack_label"], "lineage_group": r["grp"]})
            if len(picked) >= cap:
                break
        out += picked
        rep["classes"][cls] = {"available": len(d), "accepted": len(picked), "rejected": dict(rej), "groups": d.grp.nunique()}
    return out


def ingest_ids2018(srcs, cap, seed):
    """Processed CSVs have no IP columns: lineage = one campaign per day and label; no per-source window counts."""
    import pandas as pd
    s = srcs["cic_ids2018"]
    base = RAW / s["id"] / s["revision"]
    rep, frames = {"rows": 0, "unmapped": Counter(), "classes": {}}, []
    for f in s["files"]:
        day = f["path"].split("/")[-1].split("_")[0]
        df = pd.read_csv(base / f["path"], low_memory=False)
        df = df[df["Label"] != "Label"]  # repeated header rows inside the 2018 CSVs
        rep["rows"] += len(df)
        rep["unmapped"].update(df.loc[~df.Label.isin(IDS2018_MAP), "Label"].value_counts().to_dict())
        df = df[df.Label.isin(IDS2018_MAP)].rename(columns=IDS2018_RENAME)
        for c in CIC_COLS[5:]:
            if c in df and c not in ("attack_label", "Timestamp"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["protocol"] = pd.to_numeric(df["protocol"], errors="coerce").map(NUM_PROTO)
        df["destination_port"] = pd.to_numeric(df["destination_port"], errors="coerce").fillna(-1).astype(int)
        df["flow_id"] = f"{day}:" + df.index.astype(str)
        df["cls"] = df.attack_label.map(IDS2018_MAP)
        df["grp"] = f"ids2018:{day}:" + df.attack_label
        frames.append(df.dropna(subset=["Flow Duration", "Total Fwd Packets"]))
    rep["unmapped"] = dict(rep["unmapped"])
    return _pick(pd.concat(frames), s, IDS2018_MAP, cap, seed, rep), rep


def ingest_ddos2019(srcs, cap, seed):
    """Byte-prefix samples of the per-attack CSVs (pinned). Source IP + Timestamp -> per-source window counts."""
    import pandas as pd
    s = srcs["cic_ddos2019_mirror"]
    base = RAW / s["id"] / s["revision"]
    rep, frames = {"rows": 0, "unmapped": Counter(), "classes": {}}, []
    for f in s["files"]:
        df = pd.read_csv(base / f["path"], low_memory=False, on_bad_lines="skip")  # C engine: skips the cut row
        df.columns = [c.strip() for c in df.columns]
        df = df.iloc[:-1]  # the byte prefix cuts the last row
        rep["rows"] += len(df)
        rep["unmapped"].update(df.loc[~df.Label.isin(DDOS2019_MAP), "Label"].value_counts().to_dict())
        df = df.rename(columns={"Flow ID": "flow_id", "Source IP": "source_ip", "Destination IP": "destination_ip",
                                "Destination Port": "destination_port", "Protocol": "protocol", "Label": "attack_label"})
        df["protocol"] = pd.to_numeric(df["protocol"], errors="coerce").map(NUM_PROTO)
        frames.append(df)
    df = pd.concat(frames)
    df["flow_id"] = df["flow_id"].astype(str) + "@" + df["Timestamp"].astype(str)
    df = add_window_counts(df)  # over all sampled flows, before label filtering
    df = df[df.attack_label.isin(DDOS2019_MAP)].copy()
    df["cls"] = df.attack_label.map(DDOS2019_MAP)
    df["grp"] = "ddos2019:" + df.source_ip + "->" + df.destination_ip
    rep["unmapped"] = dict(rep["unmapped"])
    return _pick(df, s, DDOS2019_MAP, cap, seed, rep), rep


def ingest_dns_exf(srcs, cap, seed, window_s=60):
    """CIC-Bell-DNS-EXF-2021 stateless CSVs: one row per DNS query, no IPs, no byte counts. A record = one
    60 s window of one capture file, with measured fields only (query count, span, qname length/entropy);
    bytes stay absent (never fabricated). Lineage = the capture file (one exfil session / benign capture)."""
    import io
    import zipfile
    import pandas as pd
    s = srcs["cic_bell_dns_exf2021"]
    base = RAW / s["id"] / s["revision"]
    rep, pool = {"rows": 0, "classes": {}, "windows": Counter(), "qname_len_mean": {}, "entropy_mean": {}}, []
    for f in s["files"]:
        if not f["path"].startswith("CSV/"):
            continue  # PCAPs kept for provenance; features come from the published CSVs
        with zipfile.ZipFile(base / f["path"]) as z:
            for name in sorted(n for n in z.namelist() if "stateless_features" in n and n.endswith(".csv")):
                cls = "dns_tunneling_exfil" if "/Attacks/" in "/" + name else "benign_standard_traffic"
                df = pd.read_csv(io.BytesIO(z.read(name)))
                rep["rows"] += len(df)
                df["t"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["t"])
                df["w"] = (df["t"] - df["t"].min()).dt.total_seconds() // window_s
                grp = f"dnsexf:{f['path'].removesuffix('.zip')}:{name.split('/')[-1]}"
                for w, d in df.groupby("w"):
                    if len(d) < 2:
                        continue
                    span = (d["t"].max() - d["t"].min()).total_seconds()
                    feats = {"pkts_total": int(len(d)), "dst_port": 53, "ip_proto": 17,
                             "qname_len_mean": round(float(d["FQDN_count"].mean()), 2),
                             "qname_entropy_bits_per_char": round(float(d["entropy"].mean()), 3),
                             # repeated queries to one parent domain (taxonomy required evidence); the CSV's sld column
                             "parent_domain_top_share": round(float(d["sld"].astype(str).value_counts().iloc[0] / len(d)), 3)}
                    if span > 0:
                        feats["duration_s"] = sig(span)
                        feats["pkts_per_s"] = sig(len(d) / span)
                    pool.append({"cls": cls, "telemetry": telemetry(feats), "source": s["id"], "source_revision": s["revision"],
                                 "license_ref": s["license_spdx"], "original_id": f"{grp}:w{int(w)}",
                                 "original_label": "attack" if cls != "benign_standard_traffic" else "benign", "lineage_group": grp})
                    rep["windows"][cls] += 1
    by = {}
    for e in pool:
        by.setdefault(e["cls"], []).append(e)
    out = []
    for cls, es in by.items():  # round-robin across capture files, capped
        groups = {}
        for e in es:
            groups.setdefault(e["lineage_group"], []).append(e)
        picks, i = [], 0
        while len(picks) < cap and any(len(g) > i for g in groups.values()):
            picks += [g[i] for g in groups.values() if len(g) > i][: cap - len(picks)]
            i += 1
        out += picks
        f_ = [p["telemetry"]["flow"]["features"] for p in picks]
        rep["classes"][cls] = {"accepted": len(picks), "groups": len(groups)}
        rep["qname_len_mean"][cls] = round(sum(x["qname_len_mean"] for x in f_) / len(f_), 2)
        rep["entropy_mean"][cls] = round(sum(x["qname_entropy_bits_per_char"] for x in f_) / len(f_), 3)
    rep["windows"] = dict(rep["windows"])
    return out, rep


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

    for name, fn in (("cic_ids2018", ingest_ids2018), ("cic_ddos2019_mirror", ingest_ddos2019),
                     ("cic_bell_dns_exf2021", ingest_dns_exf)):
        if (RAW / srcs[name]["id"] / srcs[name]["revision"] / "MANIFEST.json").exists():
            got, rep = fn(srcs, cap, seed)
            pool += got
            report[name] = rep

    kept, dropped = [], Counter()
    for x in pool:
        if CONSISTENT.get(x["cls"], lambda f: True)(x["telemetry"]["flow"]["features"]):
            kept.append(x)
        else:
            dropped[f'{x["source"]}:{x["original_label"]}'] += 1
    pool = kept
    report["label_inconsistent_dropped"] = dict(dropped)
    tmp = OUT / "observed_pool.jsonl.part"
    with open(tmp, "w") as fh:
        for x in pool:
            fh.write(json.dumps(x) + "\n")
    os.replace(tmp, OUT / "observed_pool.jsonl")  # atomic: a concurrent build reads the old or the new pool, never half
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
    for src in ("cic_ids2017_mirror", "nf_ton_iot_v2_mirror", "cic_ids2018", "cic_ddos2019_mirror", "cic_bell_dns_exf2021"):
        if src in rep:
            print(src, json.dumps({c: (v["accepted"], v["groups"], v.get("rejected", {})) for c, v in rep[src]["classes"].items()}))

"""Endpoint speed check: the same records scored per record ("sequential", the evaluated path), with the cached batch
scorer ("cached", the serving default since 19 Oct) and the older batched path (several call sizes). Reports server time per record and records/s, and the largest score difference between the
two paths (they must agree).

usage: .venv/bin/python deploy/speed.py [N]      (N records from datasets_ready/fast_test.jsonl, default 96)
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.prepare import deserialize_telemetry  # noqa: E402


def run(tels, scoring, batch):
    t0, server, out = time.time(), [], []
    for i in range(0, len(tels), batch):
        res = invoke_mode(tels[i:i + batch], scoring)
        server.append(res["batch_s"])
        out += res["results"]
    return {"scoring": scoring, "batch": batch, "server_s_per_record": round(sum(server) / len(tels), 4),
            "wall_s_per_record": round((time.time() - t0) / len(tels), 4),
            "records_per_s_server": round(len(tels) / sum(server), 2)}, out


def invoke_mode(tels, scoring):
    import subprocess
    import tempfile
    from deploy.client import ENDPOINT, REGION
    with tempfile.TemporaryDirectory() as d:
        req, resp = Path(d) / "req.json", Path(d) / "resp.json"
        req.write_text(json.dumps({"records": tels, "scoring": scoring}))
        subprocess.run(["aws", "sagemaker-runtime", "invoke-endpoint", "--region", REGION, "--endpoint-name", ENDPOINT,
                        "--content-type", "application/json", "--cli-read-timeout", "70", "--body", f"fileb://{req}",
                        str(resp)], check=True, capture_output=True, text=True)
        return json.loads(resp.read_text())


def main(n=96):
    rows = [json.loads(line) for line in open("datasets_ready/fast_test.jsonl")]
    tels = [deserialize_telemetry(r["messages"][1]["content"]) for r in random.Random(3).sample(rows, n)]
    invoke_mode(tels[:2], "batched")  # warm-up (first call compiles kernels)
    report = []
    seq, ref = run(tels, "sequential", 8)
    report.append(seq)
    for scoring, b in (("cached", 8), ("cached", 16), ("batched", 8), ("batched", 16), ("batched", 32)):  # cached: the default
        r, got = run(tels, scoring, b)
        r["max_top_prob_diff_vs_sequential"] = round(max(abs(a["top"][0][1] - c["top"][0][1]) for a, c in zip(ref, got)), 5)
        r["same_top_class"] = sum(a["top"][0][0] == c["top"][0][0] for a, c in zip(ref, got))
        report.append(r)
    print(json.dumps(report, indent=1))
    Path("reports/endpoint_speed.json").write_text(json.dumps({"n": n, "runs": report, "time": time.strftime("%F %T")}, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 96)

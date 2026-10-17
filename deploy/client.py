"""Call the SageMaker endpoint with telemetry from a JSONL file (via the AWS CLI; your local credentials).

usage: .venv/bin/python deploy/client.py FILE.jsonl [--limit N] [--batch B] [--out results.jsonl]
FILE lines are telemetry objects or dataset records (messages[1] is deserialized; the label is then scored).
A real-time invocation must finish in 60 s, so records go in batches (default 8, about 1-2 s each on an L4).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.prepare import deserialize_telemetry  # noqa: E402

ENDPOINT, REGION = "qwen35-waf-v10", "us-east-1"


def invoke(records):
    with tempfile.TemporaryDirectory() as d:
        req, resp = Path(d) / "req.json", Path(d) / "resp.json"
        req.write_text(json.dumps({"records": records}))
        subprocess.run(["aws", "sagemaker-runtime", "invoke-endpoint", "--region", REGION, "--endpoint-name", ENDPOINT,
                        "--content-type", "application/json", "--accept", "application/json",
                        "--cli-read-timeout", "70", "--body", f"fileb://{req}", str(resp)],
                       check=True, capture_output=True, text=True)
        return json.loads(resp.read_text())["results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for line in open(a.file):
        r = json.loads(line)
        if "messages" in r:
            rows.append((r["id"], r["metadata"]["threat_type"], deserialize_telemetry(r["messages"][1]["content"])))
        else:
            rows.append((None, None, r))
        if len(rows) == a.limit:
            break
    results, hit, n = [], 0, 0
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        for (rid, label, _), res in zip(chunk, invoke([t for _, _, t in chunk])):
            top = res.get("top", [[None]])[0][0]
            d = res.get("decision") or {}
            if label:
                n += 1
                hit += top == label
            print(f"{rid or '-'}  label={label}  top={top}  verdict={d.get('verdict')}  "
                  f"conf={d.get('confidence')}  reason={(res.get('diagnostics') or {}).get('reason')}  {res.get('latency_s')}s")
            results.append({"id": rid, "label": label, **res})
    if n:
        print(f"argmax accuracy {hit}/{n} = {hit / n:.3f}")
    if a.out:
        Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in results))


if __name__ == "__main__":
    main()

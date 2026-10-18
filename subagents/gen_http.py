"""Scenario generation through the OpenAI-compatible proxy (no tools: the model answers in its reply, this script
validates and writes the two raw output files). Same job specs and scenario_v2 prompt as the in-session workers.

usage: set -a; . ./.env; set +a; XM_OPENAI_BASE=$base_api XM_OPENAI_KEY=$GEMINI_API_KEY \
       python -m subagents.gen_http MODEL JOB.json [JOB.json ...]      (MODEL e.g. cx/gpt-6.1-sol, ag/gemini-3.8-flash-medium)
Then: python -m subagents.orchestrator verify OUT TWIN_OUT. Raw outputs are never overwritten.
"""
import json
import os
import sys
from pathlib import Path

import jsonschema

from subagents.orchestrator import near_dup_group
from subagents.xmodel_run import chat

ROOT = Path(__file__).resolve().parent.parent
FORMAT = """
## Output format for this run (replaces "Validate before finishing")
You have no tools and cannot write files: answer in this reply only. Output exactly two sections and nothing else:
a line `### OUTPUT1` followed by exactly {count} lines (one JSON object per line, output 1), then a line
`### OUTPUT2` followed by exactly {count} benign twin lines (output 2, each with "twin_of"). No markdown fences,
no commentary. Every `telemetry` must validate against the JSON schema below; no two telemetry objects identical.

## Telemetry JSON schema
{schema}

## Record shape (one attack line and its twin from another class; do not copy their content)
{examples}

## Job
{job}
"""


def parse(text):
    secs, cur = {"1": [], "2": []}, None
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("### OUTPUT"):
            cur = s[len("### OUTPUT"):].strip()[:1]
        elif cur in secs and s.startswith("{"):
            try:
                secs[cur].append(json.loads(s))
            except json.JSONDecodeError:
                pass
    return secs["1"], secs["2"]


def run(model, job_path, tries=3):
    job = json.load(open(job_path))
    out = ROOT / job["out_path"]
    twin = ROOT / job["twin_out_path"] if job.get("twin_out_path") else None  # benign classes: no twins
    assert not out.exists() and not (twin and twin.exists()), f"refusing to overwrite {out} / {twin}"
    schema = json.load(open(ROOT / job["telemetry_schema"]))
    check = jsonschema.validators.validator_for(schema)(schema)
    ex = [open(ROOT / "data/jobs/out" / f).readline().strip() for f in
          ("rce_cmd_injection.vsn32.jsonl", "benign_standard_traffic.vsn32_rce_cmd_injection.jsonl")]
    instr = job["instructions"].split("## Validate before finishing")[0]
    spec = {k: v for k, v in job.items() if k not in ("instructions", "out_path", "twin_out_path", "telemetry_schema")}
    fmt = FORMAT if twin else FORMAT.replace("Output exactly two sections", "Output exactly one section").replace(
        ", then a line\n`### OUTPUT2` followed by exactly {count} benign twin lines (output 2, each with \"twin_of\")", "")
    prompt = instr + fmt.format(count=job["count"], schema=json.dumps(schema), examples="\n".join(ex),
                                job=json.dumps(spec, indent=1))
    why = ""
    for attempt in range(tries):
        try:
            a, t = parse(chat(model, prompt, timeout=1800))
        except Exception as e:  # network / HTTP / timeout: retried
            why = f"{type(e).__name__}: {e}"[:300]
            continue
        bad = [i for i, r in enumerate(a + t) if not isinstance(r.get("telemetry"), dict) or not check.is_valid(r["telemetry"])]
        keys = {json.dumps(r.get("telemetry"), sort_keys=True) for r in a + t}
        twin_ok = (sorted(r.get("twin_of", (r.get("scenario") or {}).get("twin_of", -1)) for r in t) == list(range(len(a)))
                   if twin else not t)
        try:  # templated output (number-only variation) collapses into few near-dup shapes: same 0.8 bar as verify
            shapes = min(len({near_dup_group(c, r["telemetry"]) for r in rs}) / max(1, len(rs))
                         for c, rs in ((job["class"], a), ("benign_standard_traffic", t)) if rs)
        except Exception:
            shapes = 0.0
        why = (f"attack {len(a)} twin {len(t)} schema_bad {len(bad)} dup {len(a) + len(t) - len(keys)} twin_of_ok {twin_ok}"
               f" distinct_shapes {shapes:.2f}")
        if len(a) == job["count"] and len(t) == (job["count"] if twin else 0) and not bad and len(keys) == len(a) + len(t) and twin_ok and shapes >= 0.8:
            out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in a))
            if twin:
                twin.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in t))
            return f"{job_path}: ok ({why})"
        print(f"{job_path} attempt {attempt + 1}: {why}", flush=True)
    return f"{job_path}: FAILED ({why})"


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(min(len(sys.argv) - 2, int(os.environ.get("XM_WORKERS", "6")))) as ex:
        for r in ex.map(lambda j: run(sys.argv[1], j), sys.argv[2:]):
            print(r, flush=True)

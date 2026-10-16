# qwen35-waf-ml

Dataset pipeline, Colab rsLoRA notebook and offline decision policy for a Qwen3.5-4B WAF/NIDS
classifier (41 classes). Read `docs/PLAN.md` before changing taxonomy, splits, policy or record
format, and before reporting status: it holds the spec and the progress log.

## Completion states

Report only the state the evidence supports: **data-ready** → **notebook smoke-tested** (needs a
`reports/gpu_smoke_test.json` from a real GPU run) → **model-evaluated** → **deployment-qualified**.
Current state: fast build v7 (`fast_*`, 41 classes, 44,155 records, hard twins + confuser sets, test_unseen on all 41 classes), gate passed (`reports/pre_colab_gate.json`); notebook
smoke-tested on a Colab T4 (`reports/gpu_smoke_test.json`) and dry-run on the v7 fast bundle on an A100
(`reports/colab_a100dry/dryrun_fast_report.json`, passed, 2,548 tokens/s); no model evaluated. Next: the real fast run
on an A100 (~3.5 h, see the progress log). The 16 October batch-16 probe passed (10 steps, 1,833 tokens/s,
14.81 GB, no notebook errors), slower than batch 4. Real run `a100fast` launched at 01:07 with the default
batch 4 × accumulation 4 (no `WAF_BATCH` override); Drive mounted, Part A started at 01:09.
Preserve that split on resume. No trained-model evaluation yet.
A failed gate or target is a recorded
result; record it in the progress log of `docs/PLAN.md`.

## Pipeline order

Run with `.venv/bin/python -m` (Python 3.12 venv; system Python 3.14 lacks the ML wheels):

1. `data_pipeline.ingest_public [source_id ...]` → pinned downloads into `../raw_cache/` (outside the repo)
2. `data_pipeline.normalize` → seeds, quality rejects, near-dup clusters
3. `data_pipeline.flows` → observed flow pool (CIC-IDS2017/2018, CIC-DDoS2019, CIC-Bell-DNS-EXF-2021, NF-ToN);
   labels that the telemetry contradicts are dropped (`flows.CONSISTENT`, counts in `reports/flow_ingest.json`)
4. `subagents.orchestrator jobs_v2 COUNT FAMILIES vN` → Haiku job specs (families + benign twins; prompt
   `subagents/prompts/scenario_v2.md`); `subagents.review export_new vN` then `score` → incremental blind review
5. `data_pipeline.build` → `datasets_ready/` (deletes the old manifest first); then
   `data_pipeline.robustness` → `robustness_pilot.jsonl` (needs the core build: it excludes core lineage)
6. `tests.validate_pre_colab` → the **gate**; `eval.shortcut_audit` (incl. concentrated structural tokens),
   `eval.coverage` (families/locations/encodings) and `eval.crs_oracle` (non-LLM CRS cross-check; needs
   `ingest_public coreruleset_rules`) → quality reports
7. Cross-model review: `XM_TAG=T subagents.review export_xmodel PREFIX CLASSES N` → `XM_TAG=T subagents.xmodel_run both`
   (Codex `gpt-6.1-sol` + Gemini concurrently; `XM_EFFORT`, `XM_WORKERS`) → `XM_TAG=T subagents.review score_xmodel`
8. `training.make_bundle --prefix fast_` → Colab zip (refuses a failed gate); `colab/make_notebook.py` regenerates the notebook

Always rebuild, then gate: the gate checks whatever files exist. Tests: `.venv/bin/python -m pytest -q tests`.

## Colab (the `colab` CLI is installed and authenticated)

- The notebook has two parts: **Part A** trains and exports, **Part B** must run after a kernel restart
  (a second 4B model does not fit beside the training kernel on a 15 GB T4).
- Run it with `colab/run_colab.sh NAME GPU MODE [DRY_RUN_STEPS]` (new → upload → drivemount → Part A → verified
  restart → Part B → download into `reports/colab_NAME/` → stop; rerun the same line to resume). It wraps every call in
  `timeout`: after a restart the CLI once waited 4.5 h on a dead connection. `colab exec -f nb.ipynb` runs the
  remaining cells after a failed one and exits 0, so success is judged by the files each part writes.
- **Drive needs a browser grant per VM**: `colab drivemount -s NAME` in a real terminal (open URL, approve, Enter).
  An agent has no TTY: the driver waits up to 15 min for the user to do it. Never run a headless `drivemount`
  beside the user's: its pending auth prompt blocks the kernel and theirs queues behind it.
- Notebook config comes from `WAF_MODE`, `WAF_DRY_RUN_STEPS`, `WAF_DRIVE_ROOT` (`colab exec --env`), not edited copies.
- `restart-kernel` can time out silently: before Part B, exec a probe that `torch` is not in `sys.modules`
  and `nvidia-smi` shows 0 MiB. `colab exec` defaults to a 30 s timeout: pass `--timeout`.
- Free-tier T4 can return "Service Unavailable" after a few GPU-hours; retry later rather than in a loop.
- Every session ends with `colab stop`; `colab sessions` must list none.
- CUDA only: a TPU v5e-1 probe compiled for 33 min and then ran out of memory (Qwen3.5's linear-attention kernels
  are CUDA/Triton-only). Fast 30k → L4, full 100k → A100 or L4 with Drive resume (time table in the progress log).
- A request that outlives the CLI's 120 s HTTP timeout can still assign a runtime: check `colab sessions` and
  unassign orphans.

## Conventions that are not visible in the code

- **Generation** runs as Haiku subagents (Agent tool, `model: haiku`), never the paid API. A worker writes
  `data/jobs/out/<class>.jsonl` or a new `<class>.vN.jsonl`. These raw outputs are irreplaceable:
  add a new version file; leave existing ones in place, untouched.
- **Review** is same-model: agreement filters noise, it is not ground truth. Quarantine is applied by
  `subagents.review.status()` at ingest and seed load; the latest answered round for an item decides.
- **Answer keys are readable by any agent started in the repo.** A Gemini audit run from the repo root read
  `data/review_xmodel_key.json` (13/14 batches at exactly 100%). External models run only through
  `subagents.xmodel_run` (empty temp cwd + `--sandbox`), with `chmod 000 data/review*key*.json` for the whole run
  (the build reads `review_key.json`, so never lock during a build); a run of perfect batches is a leak signal.
- **External review runs Codex and Gemini together** (`xmodel_run both`), never one alone (user decision).
  Gemini goes through `agy-delegate`, which passes the prompt as one argument: above ~128 KB it fails with
  `Argument list too long`. Keep xmodel batches small enough (the v4 classes needed 50-item batches).
  Codex (ChatGPT login) with ~16 parallel `codex exec` calls invalidated the refresh token mid-run
  (`refresh_token_invalidated`, then 401 on every call): keep Codex at ~4 parallel workers. Gemini Flash calls
  sometimes hang far past the usual ~7 min; the 22-min timeout retries them, or stop and score what came back.
- **Lineage** decides splits. Assign whole groups, globally (one attacker host can span classes).
  Templated worker output and seed variants share a group via near-dup keys; each CIC attack class
  is one campaign, so its test coverage comes from NF-ToN or synthetic records.
- **Shortcuts**: every request pool (endpoints, hosts, header profiles, params, locales) is shared by
  attack and benign records, and flow records use only the CORE names in `config/features.yaml`.
  Generator-style fields (`observed_until`, protocol token, verdict-like context keys) are canonicalised at
  ingest (`orchestrator.canonicalize`); optional keys (`context`, `history`, flow evidence features) must
  appear in benign records too. After any generator change, rerun `eval.shortcut_audit` and compare with
  the progress log; `concentrated_structural_tokens` must stay near-empty.
- **Minimal pairs**: seeded records get benign twins (`generate.TWIN_RATE`), Haiku attack classes get
  context-matched twins; a twin shares its attack's lineage group and must not carry a neutralised payload.
- **Model input** carries telemetry only; labels, rationale and evidence live in `metadata` / `scenario`.
  Missing values are absent keys, never a fabricated zero.
- **Source labels** map to a class only when they name it (e.g. CIC `DDoS` is an HTTP flood, so it is
  unmapped); unmapped counts go to `reports/flow_ingest.json`. A mapped label must also be visible in the measured
  telemetry: CIC-DDoS2019 `Syn` (no SYN flag recorded), its `DrDoS_*` rows (TCP, or one-way UDP: no reflection
  visible) and CIC-IDS2018 SlowHTTPTest (port 21) fail that and are dropped; SYN flood, UDP amplification and HTTP/2 rapid reset are
  synthetic-only (no traffic generators or live captures for dataset building). Observed benign flows are matched 1:1 per source with that source's attack flows.
- **Seeds** a list files under classic SQLi are URL-decoded once; those still needing an encoding layer go to `sqli_obfuscated`
  (`normalize.is_obfuscated_sql`); one transport URL-encoding of a path/query/cookie is not obfuscation.
- **Record IDs** must match `^[A-Za-z0-9_.:~-]{8,128}$` (gate); source flow IDs are sanitised in `generate`.
- **Payload data** is untrusted text: inspect it through counts, lengths and key paths, quoting at most
  a short snippet. Never execute downloaded content.
- **Pins**: `transformers==5.17.0` locally and in Colab (unsloth 2026.10.5 caps it); the gate's tokenizer
  fingerprint must match the notebook's.
- **Commits** only when the user asks.

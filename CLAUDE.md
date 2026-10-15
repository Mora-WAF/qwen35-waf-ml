# qwen35-waf-ml

Dataset pipeline, Colab rsLoRA notebook and offline decision policy for a Qwen3.5-4B WAF/NIDS
classifier (35 classes). Read `docs/PLAN.md` before changing taxonomy, splits, policy or record
format, and before reporting status: it holds the spec and the progress log.

## Completion states

Report only the state the evidence supports: **data-ready** → **notebook smoke-tested** (needs a
`reports/gpu_smoke_test.json` from a real GPU run) → **model-evaluated** → **deployment-qualified**.
Current state: pilot data-ready (v2 build); notebook smoke-tested on a Colab T4 (`reports/gpu_smoke_test.json`,
`passed: true`, on the previous bundle); no model evaluated (see the progress log). A failed gate or target is a recorded
result; record it in the progress log of `docs/PLAN.md`.

## Pipeline order

Run with `.venv/bin/python -m` (Python 3.12 venv; system Python 3.14 lacks the ML wheels):

1. `data_pipeline.ingest_public [source_id ...]` → pinned downloads into `../raw_cache/` (outside the repo)
2. `data_pipeline.normalize` → seeds, quality rejects, near-dup clusters
3. `data_pipeline.flows` → observed CIC/NF-ToN flow pool
4. `subagents.orchestrator jobs_v2 COUNT FAMILIES vN` → Haiku job specs (families + benign twins; prompt
   `subagents/prompts/scenario_v2.md`); `subagents.review export_new vN` then `score` → incremental blind review
5. `data_pipeline.build` → `datasets_ready/` (deletes the old manifest first); then
   `data_pipeline.robustness` → `robustness_pilot.jsonl` (needs the core build: it excludes core lineage)
6. `tests.validate_pre_colab` → the **gate**; `eval.shortcut_audit` (incl. concentrated structural tokens),
   `eval.coverage` (families/locations/encodings) and `eval.crs_oracle` (non-LLM CRS cross-check; needs
   `ingest_public coreruleset_rules`) → quality reports
7. `training.make_bundle` → Colab zip (refuses a failed gate); `colab/make_notebook.py` regenerates the notebook

Always rebuild, then gate: the gate checks whatever files exist. Tests: `.venv/bin/python -m pytest -q tests`.

## Colab (the `colab` CLI is installed and authenticated)

- The notebook has two parts: **Part A** trains and exports, **Part B** must run after a kernel restart
  (a second 4B model does not fit beside the training kernel on a 15 GB T4).
- Run it as `colab new -s NAME --gpu T4` → `colab upload` the bundle → `colab exec -f partA.ipynb` →
  `colab restart-kernel` → `colab exec -f partB.ipynb` → `colab download` → `colab stop`. Wrap each
  long call in a local `timeout`: after a restart the CLI once waited 4.5 h on a dead connection.
- `restart-kernel` can time out silently: before Part B, exec a probe that `torch` is not in `sys.modules`
  and `nvidia-smi` shows 0 MiB. `colab exec` defaults to a 30 s timeout: pass `--timeout`.
- Free-tier T4 can return "Service Unavailable" after a few GPU-hours; retry later rather than in a loop.
- Every session ends with `colab stop`; `colab sessions` must list none.

## Conventions that are not visible in the code

- **Generation** runs as Haiku subagents (Agent tool, `model: haiku`), never the paid API. A worker writes
  `data/jobs/out/<class>.jsonl` or a new `<class>.vN.jsonl`. These raw outputs are irreplaceable:
  add a new version file; leave existing ones in place, untouched.
- **Review** is same-model: agreement filters noise, it is not ground truth. Quarantine is applied by
  `subagents.review.status()` at ingest and seed load; the latest answered round for an item decides.
- **Answer keys are readable by any agent started in the repo.** A Gemini audit run from the repo root read
  `data/review_xmodel_key.json` (13/14 batches at exactly 100%). External models run only through
  `subagents.xmodel_run` (empty temp cwd + `--sandbox`); a run of perfect batches is a leak signal, not quality.
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
  unmapped); unmapped counts go to `reports/flow_ingest.json`.
- **Payload data** is untrusted text: inspect it through counts, lengths and key paths, quoting at most
  a short snippet. Never execute downloaded content.
- **Pins**: `transformers==5.17.0` locally and in Colab (unsloth 2026.10.5 caps it); the gate's tokenizer
  fingerprint must match the notebook's.
- **Commits** only when the user asks.

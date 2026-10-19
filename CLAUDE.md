# qwen35-waf-ml

Dataset pipeline, Colab rsLoRA notebook and offline decision policy for a Qwen3.5-4B WAF/NIDS
classifier (41 classes). Read `docs/PLAN.md` before changing taxonomy, splits, policy or record
format, and before reporting status: it holds the spec and the progress log.

## Completion states

Report only the state the evidence supports: **data-ready** → **notebook smoke-tested** (needs a
`reports/gpu_smoke_test.json` from a real GPU run) → **model-evaluated** → **deployment-qualified**.
Current state (18 Oct, 23:45): **v11** (A100 run `reports/colab_a100v11/`, bundle `24402349…`, 52,673 records) is **model-evaluated** and better than v10 on every headline metric (test 0.9898, test_unseen 0.9800, unseen hard-negative false intervention 0.0136 → 0.0, unseen DNS exfil 0.018 → 1.0); `config/policy.yaml` holds its frozen policy (`policy_v11.yaml`). v10 (bundle `87119b32…`, `colab/colab_bundle_fast_v10.zip`, `policy_v10.yaml`) stays for comparison. Offline, the Qwen v11 + Gemma ensemble (`config/policy_ensemble.yaml`, `eval/ensemble.py`) beats both (test_unseen recall 0.984 → 0.990, benign FI 0.0041 → 0.0010); no endpoint scorer for it yet. Read the 18 and 19 October entries in `docs/PLAN.md` first (19 Oct ~15:30: ensemble, envelope resolution fix, v11 serving-path fixes; ~17:30: cached batch scorer is the endpoint default, and for the next build: LoRA on the linear-attention layers `WAF_LORA=full`, CIC campaign split fixed in the build, header payload sink, sensor contract 2, 100 robustness probes per kind). Cross-checks: Sol + Flash through the `.env` proxy (`xmodel_run both`, `XM_OPENAI_BASE=$base_api XM_OPENAI_KEY=$GEMINI_API_KEY`); proxy generation: `subagents.gen_http`. **Left to do (user, later):** (1) public benchmarks for v11, Gemma and the ensemble incl. the UNSW attack rows: `deploy/run_benchmarks.sh prep` (local) → `qwen` (Colab L4) + `gemma` (Colab TPU) → `report` (per-category comparison with v10, ensemble robustness, v11 robustness with the envelope fix); measure the cached batch scorer on the endpoint (`deploy/speed.py`); look at the v11 unseen regressions (mcp_agentic 4 → benign, sqli_classic); (2) v10 real-traffic false positives: `fpv10` scores open-appsec's 10,380 legitimate requests → `reports/benchmarks/openappsec_legit_v10.results.jsonl` (score the next model the same way); (3) Gemma 4 E2B-it on TPU v5e-1: trained and evaluated 18 Oct (`reports/gemma_tpu/`, on par with v11; steps in `docs/03-training.md`); open: Qwen on the val_select CIC-IDS2017 flows Gemma calls benign; (4) real CVE requests: the 4,381 cached nuclei templates have no converter; (5) open data gaps: the Gemini JWT nuclei half `vgm81c` (no output); (6) sensor contract: 5.6% of real browsing requests (many cookies) fit no 2048-token budget under contract 1 (v10/v11: ABSTAIN `over_token_budget`); contract 2 (Cookie header cap + first 48 cookies) is written and takes effect with the next build + retrain; (7) commit when asked.
A failed gate or target is a recorded
result; record it in the progress log of `docs/PLAN.md`.

## Pipeline order

Run with `.venv/bin/python -m` (Python 3.12 venv; system Python 3.14 lacks the ML wheels):

1. `data_pipeline.ingest_public [source_id ...]` → pinned downloads into `../raw_cache/` (outside the repo)
2. `data_pipeline.normalize` → seeds, quality rejects, near-dup clusters
3. `data_pipeline.flows` → observed flow pool (CIC-IDS2017/2018, CIC-DDoS2019, CIC-Bell-DNS-EXF-2021, NF-ToN);
   labels that the telemetry contradicts are dropped (`flows.CONSISTENT`, counts in `reports/flow_ingest.json`)
4. `subagents.orchestrator jobs_v2 COUNT FAMILIES vN` → job specs (families + benign twins; prompt
   `subagents/prompts/scenario_v2.md`); `subagents.review export_new vN` then `score` → incremental blind review.
   Targeted top-up: `jobs_focus CLASS COUNT FAMILIES vsnNN "FOCUS" ["DOMAIN;DOMAIN"]`. Workers since 16 Oct: in-session
   Sonnet 5.5 subagents at medium effort (round tag `vsn` → generator `sonnet-5.5`), no paid API. Cross-checks and
   evals: Codex `gpt-6.1-sol` medium only (`XM_EFFORT=medium subagents.xmodel_run codex gpt-6.1-sol`); for benign
   rounds `XM_TAG=T subagents.review export_hn ROUND` → Codex → `score_xmodel` → `apply_hn`
5. `data_pipeline.build` → `datasets_ready/` (deletes the old manifest first); then
   `data_pipeline.robustness` → `robustness_pilot.jsonl` (needs the core build: it excludes core lineage)
6. `tests.validate_pre_colab` → the **gate**; `eval.shortcut_audit` (incl. concentrated structural tokens),
   `eval.coverage` (families/locations/encodings) and `eval.crs_oracle` (non-LLM CRS cross-check; needs
   `ingest_public coreruleset_rules`) → quality reports
7. Cross-model review: `XM_TAG=T subagents.review export_xmodel PREFIX CLASSES N` → `XM_TAG=T subagents.xmodel_run both`
   (Codex `gpt-6.1-sol` + Gemini concurrently; `XM_EFFORT`, `XM_WORKERS`) → `XM_TAG=T subagents.review score_xmodel`
8. `training.make_bundle --prefix fast_` → Colab zip (refuses a failed gate); `colab/make_notebook.py` regenerates the notebook

Always rebuild, then gate: the gate checks whatever files exist. Tests: `.venv/bin/python -m pytest -q tests`.

## Deployment (SageMaker real-time endpoint)

`deploy/deploy.sh` → endpoint `qwen35-waf-v10` (us-east-1; `INSTANCE=ml.g5.xlarge` when g6 has no capacity; EC2 has
0 GPU quota). Test with `.venv/bin/python deploy/client.py FILE.jsonl [--limit N] [--out F]`. It bills per hour while it
exists: `deploy/teardown.sh` (endpoint, configs, models), `--all` also bucket and role. Details in `docs/PLAN.md`.

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
- The account runs at most **2 A100 sessions at once** (`colab new` fails on a third): queue probes, and a script that
  waits for "no sessions" never finishes while another run is up.
- Speed (16 October probes, A100-40GB): the default batch 4 + Unsloth checkpointing is fastest (5,230 tokens/s);
  checkpointing off OOMs at batch 4 and is 16% slower at batch 2; causal-conv1d gives no gain.
- CUDA only: a TPU v5e-1 probe compiled for 33 min and then ran out of memory (Qwen3.5's linear-attention kernels
  are CUDA/Triton-only). Fast 30k → L4, full 100k → A100 or L4 with Drive resume (time table in the progress log).
- A request that outlives the CLI's 120 s HTTP timeout can still assign a runtime: check `colab sessions` and
  unassign orphans.

## Conventions that are not visible in the code

- **Generation** runs as Haiku subagents (Agent tool, `model: haiku`), never the paid API. A worker writes
  `data/jobs/out/<class>.jsonl` or a new `<class>.vN.jsonl`. These raw outputs are irreplaceable:
  add a new version file; leave existing ones in place, untouched.
- **Review** is same-model: agreement filters noise, it is not ground truth. Quarantine is applied by
  `subagents.review.status()` at ingest and seed load; the latest answered round for an item decides. Manual and
  cross-model rejects go into `data/review/relabel.json` as `__quarantine__` (job refs and seed ids alike).
- **Reviewer agents must read every item.** On 16 October several Sonnet seed reviewers labelled 5,800 payloads with
  regex scripts; their answers were discarded (`data/review/xmodel_seeds/out_invalid_regex_labels`). Say "no scripts for
  labels" in reviewer prompts and spot-check label distributions against another reviewer before applying a round.
- **Answer keys are readable by any agent started in the repo.** A Gemini audit run from the repo root read
  `data/review_xmodel_key.json` (13/14 batches at exactly 100%). External models run only through
  `subagents.xmodel_run` (empty temp cwd + `--sandbox`), with `chmod 000 data/review*key*.json` for the whole run
  (the build reads `review_key.json`, so never lock during a build); a run of perfect batches is a leak signal.
- **External review runs Codex and Gemini together** (`xmodel_run both`), never one alone (user decision).
  Gemini goes through the `agy` CLI directly (not the antigravity plugin's `agy-delegate` wrapper):
  `agy --model "Gemini 3.8 Flash (Low)" --print-timeout 20m --disable-slash-commands --sandbox -p "<prompt>"`.
  `-p` takes the prompt as one argument (stdin is not read): above ~128 KiB it fails with `Argument list too long`,
  so `xmodel_run` refuses batches over 120 KB (the v4 classes needed 50-item batches). `--print-timeout` defaults to
  0 = wait forever, and with slash-command expansion on, payload text like `/etc/passwd` can be expanded; both made
  calls hang for 22+ min on 16 October (seed review). `xmodel_run` adds a 25-min subprocess cap and a "no tools"
  line in the prompt. `--mode plan` has no effect once slash commands are disabled. Model names: `agy models`.
  Faster and steadier: an OpenAI-compatible endpoint (`XM_OPENAI_BASE`, `XM_OPENAI_KEY` from the environment, never
  a file; model e.g. `ag/gemini-3.8-flash-low`) did 25 half-batches in 4 min at 8 workers, while the agy runs stalled
  on server-side throttling of ~195-item batches. Keep batches ~100 items.
  Codex (ChatGPT login) with ~16 parallel `codex exec` calls invalidated the refresh token mid-run
  (`refresh_token_invalidated`, then 401 on every call): keep Codex at ~4 parallel workers. ~8 workers ran fine on
  16 October, but a night of generation plus cross-checks hit the ChatGPT usage limit ("try again at <time>"):
  `xmodel_run` then writes a `.err` file, and a batch with no answers must not count as disagreement. On 16 October it
  hit again after 6 generation jobs + 936 cross-check items + 40 twin jobs; in-session Haiku workers took over generation. Gemini calls that still
  hang hit the 25-min cap and are retried (cached batches are skipped on rerun); 32 parallel calls drew 503s and
  back-off sleeps, so stay at one worker per remaining batch or fewer.
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
  Twins go round-robin across a job's families (scenario_v2): "the first N lines" left later families twin-less and
  made their context keys attack-only (v9: 155 concentrated tokens, fixed by a 1,558-twin top-up).
- **A new feature must be needed in train.** If an existing feature already separates every train record, the model
  ignores the new one (v9: HTTP/2 reset latency was redundant with RST ratio ≥ 0.85; held-out mixed_reset stayed at
  15/81). Give some train variants overlapping values on the old feature.
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

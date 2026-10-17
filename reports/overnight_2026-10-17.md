# Overnight report — 17 October 2026

Scope agreed before: no training, no rebuild of `datasets_ready/` (it holds the v10 data), benchmarks allowed on
Colab and SageMaker. Every change takes effect at the next build. Tests: 124 pass.

## Gaps closed in code (training + inference use the same functions in `training/prepare.py`)

| Gap | Fix | Check |
|---|---|---|
| JWT duplicate claim hidden in base64 | `add_decoded_views`: a `jwt` view (header.payload JSON text as encoded, duplicates kept) | test `test_decoded_views_jwt_base64_and_limits` |
| Base64 payloads (GoTestWAF 21% blocked) | `base64` views for printable decodes (binary IDs skipped; query, cookie, header, form, JSON, multipart, path); generator `B64_RATE` 6% Base64 layer for attacks (twins inherit) and benign alike | `test_base64_layer_round_trips_through_the_sensor_view` |
| Size-padding evasion | `collapse_runs` + `budget_windows`: the 128 KB padded GoTestWAF SQLi becomes one 525-token prompt with the payload visible | `test_budget_windows_collapse_padding_and_window_long_bodies` |
| 45% of real requests over budget | one `fit_budget` (sensor contract); 4% of training HTTP records carry the sensor's truncation (`truncation_augment`) | `test_truncation_augment_is_label_independent_and_well_formed` |
| NetFlow v1 without packet lengths | synthetic flows drop them at 15% (`NO_PKT_LEN_RATE`) | flow-consistency test updated |
| Envelope bound read as a string | `eval.calibration` writes YAML-safe floats | `test_envelope_written_by_calibration_reloads_as_numbers` |
| Colab Drive hang | `run_colab.sh`: kernel round trip required after the mount | — |
| SageMaker capacity | `deploy.sh`: instance-type fallback, rolled-back update counted as a failure, `set -e`-safe wait | used tonight (g6 had no capacity twice; g5 deployed) |

## New data (raw outputs, Sonnet 5.5 medium; Codex `gpt-6.1-sol` medium + Gemini 3.8 Flash Medium blind review)

| Round | Records | Review |
|---|---|---|
| `vsn21`–`vsn23` benign search / free text with shell or SQL words (GoTestWAF false positives) | 180 | both: 180/180 benign; 17 both called standard traffic → relabelled |
| `vsn31` LDAP/XPath without the SQL-style tautology | 40 + 40 twins | both confirm every attack; a perfect-separator flag (`input_handling.escaping_applied`) removed before review |
| `vsn32` SSI exec directives (`rce_cmd_injection`) | 40 + 40 twins | both confirm every attack; 1 twin relabelled (both reviewers) |

With yesterday's rounds: 858 Sonnet records ingest cleanly.

## Decisions taken (reversible, recorded in `docs/PLAN.md`)

- Scanner User-Agents: no model class; a tool name is not attack evidence alone → sensor tier-0 rule.
- Log4Shell: `ssti` scope widened to expression-language / message-lookup injection (CWE-917). No training round
  run for it yet.
- SSI: `rce_cmd_injection` (scope text updated).
- Out-of-taxonomy abstention (64%): not fixable by threshold or envelope; the probes that pass (Bot, Infiltration)
  sit inside the benign range on every flow feature. Needs host/session-level telemetry.

## Measurements

- Endpoint (`ml.g5.xlarge`, A10G), `deploy/speed.py`, 96 records: per-record 0.27 s/record; the new batched scorer
  0.71–1.06 s/record (slower: it recomputes the prompt per kept class). Default switched back to per-record. Batched
  matches the A100 reference within 0.011 probability (8/8 top class); per-record within 0.036 (7/8, one near tie
  between two attack classes). `fla` kernel loads; `causal-conv1d` wheel fails on an ABI mismatch.
- Colab L4, 400 test records: HF per-record 3.56 records/s, HF batched 0.75 (confirms the endpoint result).
  vLLM 0.30 with single-token scoring requests was slower still: for Qwen3.5 its prefix cache uses 528-token blocks
  and only reuses full blocks, so ~400-token prompts are recomputed for every request; stopped before it finished.
  A generation-based vLLM scorer (`vllm_bench.py vllm_gen`: one greedy generation per record with top-20 log-probs,
  class scores rebuilt from it) is written but not run (stopped at the user's request, 09:40).

## CVE coverage (morning, metadata only)

`eval/kev_coverage.py` → `reports/kev_coverage.json`: of 1,748 KEV CVEs, 30% map to classes detectable from the
request, 15% need gateway context, 28% are not request-visible, 27% unmapped. Thinnest classes per KEV demand:
rce_cmd_injection, deserialization, ssti/code injection, path_traversal_lfi. CVE-specific exploit data was not
generated in this session.

## Problems tonight

- The work stalled from about 02:30 to 08:30: two background jobs finished, and nothing was picked up until the
  user's message. The endpoint stayed up idle for about 6 hours (roughly $8–9). All AWS resources are deleted now.
- Colab uploads over ~10 MB fail with HTTP 500 (8 MB parts + checksum now); long CLI calls lose the connection
  (stages now run detached and are polled); Colab's torchao 0.10 breaks peft (removed in the install step).

## State at stop (09:40)

No AWS resources (endpoint, configs, models, bucket, role, log group all deleted), no Colab session, no local jobs.

## Still open

- vLLM throughput result; a cached-prefix batched scorer; `causal-conv1d` built for the image's ABI.
- Log4Shell / expression-language training data.
- Finishing the stopped benchmarks (UNSW, open-appsec) — meaningful mostly for the next model.
- Train the next version (all fixes above take effect only then).

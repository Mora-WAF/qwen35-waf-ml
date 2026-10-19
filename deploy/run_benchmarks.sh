#!/usr/bin/env bash
# Public benchmarks (GoTestWAF, open-appsec attacks + legitimate, NF-UNSW-NB15 incl. its attack rows) for Qwen v11,
# Gemma 4 E2B and their ensemble, compared per category with v10. No training. Stages, in order:
#   prep    (this machine, CPU)  training overlap vs the v11 build, one Qwen input for every benchmark record, the Gemma
#                                export of the same records + the robustness set (Gemma tokenizer from the Hub)
#   qwen    (Colab L4, ~2-3 h)   the endpoint's predict_fn with the v11 package (decoded views per serving.json) on every
#                                record, calibrated distributions included -> NAME.v11.results.jsonl
#   gemma   (Colab TPU, ~1.5 h)  the trained Gemma adapter scores the exported windows -> reports/gemma_tpu/colab_bench/
#   report  (this machine)       Gemma results, ensemble results, re-decisions, per-tag reports, comparisons, the
#                                ensemble's robustness, v11 robustness with the envelope fix
#   usage: deploy/run_benchmarks.sh prep|qwen|gemma|report      (qwen and gemma can run at the same time)
# Needs: datasets_ready/ = the v11 build (fast_train.jsonl, robustness_pilot.jsonl), colab/colab_bundle_fast.zip = the
# v11 bundle, reports/benchmarks/openappsec.jsonl (local only: captured traffic with AWS keys), the colab CLI.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python B=reports/benchmarks
FP_IN=$B/bench_in.jsonl.gz FP_OUT=$B/bench_v11.fp_out.jsonl GEMMA_TGZ=reports/gemma_tpu/colab/gemma_data_bench.tgz
say() { echo "[run_benchmarks $(date +%T)] $*"; }
case "${1:-}" in
prep)
  for n in gotestwaf openappsec unsw; do [ -f "$B/$n.jsonl" ] || { say "missing $B/$n.jsonl (eval.public_bench build $n)"; exit 1; }; done
  $PY -m eval.public_bench retag datasets_ready/fast_train.jsonl --key in_train_v11
  $PY -m eval.public_bench pack gotestwaf openappsec unsw --out "$FP_IN"
  $PY -m training.gemma_tpu.data bench --names gotestwaf,openappsec,unsw --robustness
  tar -czf "$GEMMA_TGZ" -C reports/gemma_tpu/data_bench .
  say "prep done: $FP_IN, $GEMMA_TGZ ($(du -h "$GEMMA_TGZ" | cut -f1))";;
qwen)
  [ -f "$FP_IN" ] || { say "run prep first"; exit 1; }
  RUN_DIR=reports/colab_a100v11 BUNDLE_ZIP=colab/colab_bundle_fast.zip FP_IN=$FP_IN FP_OUT=$FP_OUT \
    STAGES="merge:2400 fp:21600" colab/run_vllm_bench.sh benchv11 L4
  $PY -m eval.public_bench unpack "$FP_OUT" --tag v11;;
gemma)
  [ -f "$GEMMA_TGZ" ] || { say "run prep first"; exit 1; }
  GEMMA_SCORE="bench robustness" GEMMA_LORA=reports/gemma_tpu/colab_full/lora.npz \
    training/gemma_tpu/run_tpu.sh bench "$GEMMA_TGZ";;
report)
  G=reports/gemma_tpu/colab_bench
  $PY -m eval.public_bench gemma --scores $G/SCORES_bench.jsonl --manifest reports/gemma_tpu/data_bench/bench_manifest.json \
    --policy config/policy_gemma.yaml --as gemma
  $PY -m eval.public_bench ensemble --policy config/policy_ensemble.yaml --members v11,gemma --as ensemble
  # v10 under today's decision wrapper (envelope resolution rule): its results carry no distributions, top_class view
  $PY -m eval.public_bench redecide --src v10 \
    --policy reports/colab_a100v10/policy_v10.yaml --as v10_envfix --scored-temperature 1.083018
  for t in v11 gemma ensemble; do $PY -m eval.public_bench report --tag $t; done
  $PY -m eval.public_bench compare --tags v10,v11,gemma,ensemble --compare-out reports/benchmarks_compare.json
  $PY -m eval.public_bench compare --tags v11,gemma,ensemble --compare-out reports/benchmarks_compare_v11.json
  # ensemble robustness (the Gemma robustness scores join the member's score file) and v11 robustness, envelope fix
  $PY -m eval.ensemble build --member qwen35_4b_v11 reports/colab_a100v11/SCORES_fast.jsonl reports/colab_a100v11/policy_v11.yaml \
    --member gemma4_e2b_it_1ep reports/gemma_tpu/SCORES_fast.jsonl,$G/SCORES_robustness.jsonl config/policy_gemma.yaml
  $PY -m eval.calibration reports/colab_a100v11/SCORES_fast.jsonl --out reports/calibration_v11_envfix.json >/dev/null
  $PY - <<'EOF'
import json
r = json.load(open("reports/calibration_v11_envfix.json"))
for k in ("robustness", "robustness_exact_envelope"):
    print(k, {kind: v["pass"] for kind, v in r[k]["by_kind"].items()})
EOF
  say "report done: reports/benchmarks_compare.json, reports/benchmarks_compare_v11.json, reports/ensemble/REPORT.json";;
*) sed -n 2,16p "$0"; exit 1;;
esac

#!/usr/bin/env bash
# One notebook run through the colab CLI:
#   new -> upload bundle -> drivemount -> Part A -> verified kernel restart -> Part B -> download -> stop
# usage: colab/run_colab.sh NAME GPU MODE [DRY_RUN_STEPS]
#   colab/run_colab.sh a100dry A100 fast 25     # dry run
#   colab/run_colab.sh a100fast A100 fast       # real run; rerun the same line after a disconnect to resume from Drive
# Env: NO_DRIVE=1, WAF_BATCH=N, PART_A_ONLY=1 (throughput/memory probe: stop after Part A).
# Drive needs one browser grant per VM (Colab asks again for every new runtime): run this from a terminal and open the
# printed URL, or from anywhere and run `colab drivemount -s NAME` in a terminal when asked. NO_DRIVE=1 skips Drive.
# `colab exec` runs later cells after a failed one and still exits 0: success is judged by the files each part writes.
set -uo pipefail
NAME=$1 GPU=$2 MODE=$3 DRY=${4:-}
cd "$(dirname "$0")"
OUT=../reports/colab_$NAME; mkdir -p "$OUT"
LOG="$OUT/driver.log"
say() { echo "[run_colab $(date +%T)] $*" | tee -a "$LOG"; }
die() { say "FAILED: $*"; exit 1; }
CELL_S=$([ -n "$DRY" ] && echo 7200 || echo 86400)    # per-cell timeout: the training cell is one cell
ENV=(--env "WAF_MODE=$MODE"); [ -n "$DRY" ] && ENV+=(--env "WAF_DRY_RUN_STEPS=$DRY")
[ -n "${NO_DRIVE:-}" ] && ENV+=(--env "WAF_DRIVE_ROOT=")
[ -n "${WAF_BATCH:-}" ] && ENV+=(--env "WAF_BATCH=$WAF_BATCH")   # per-device batch override (must divide 16)
for v in WAF_GC WAF_CCONV WAF_EVAL_EVERY; do [ -n "${!v:-}" ] && ENV+=(--env "$v=${!v}"); done  # speed-probe switches
REPORT=$([ "$MODE" = smoke ] && echo gpu_smoke_test.json || { [ -n "$DRY" ] && echo "dryrun_${MODE}_report.json" || echo "train_${MODE}_report.json"; })

# Stop the VM on any exit, except when Part A is still training remotely (then the kernel keeps going; rerun to collect).
KEEP=0
trap '[ $KEEP = 1 ] && say "session $NAME left RUNNING (training in progress)" || { timeout 180 colab stop -s "$NAME" >>"$LOG" 2>&1; colab sessions 2>&1 | tee -a "$LOG"; }' EXIT

fetch() { timeout 600 colab download -s "$NAME" "/content/$1" "$OUT/$1" >>"$LOG" 2>&1 && [ -s "$OUT/$1" ]; }
busy() { colab status -s "$NAME" 2>&1 | grep -q BUSY; }
pyexec() { timeout $(( ${2:-120} + 60 )) colab exec -s "$NAME" --timeout "${2:-120}" 2>&1 <<<"$1" | tee -a "$LOG"; }

if colab sessions 2>/dev/null | grep -q "\b$NAME\b"; then
  say "reusing session $NAME"
else
  say "new $GPU session $NAME"
  timeout 900 colab new -s "$NAME" --gpu "$GPU" >>"$LOG" 2>&1 || { colab sessions | tee -a "$LOG"; die "colab new (orphan runtimes are listed above)"; }
fi
pyexec 'import subprocess; print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],capture_output=True,text=True).stdout)'

if ! fetch part_a.json && ! busy; then
  say "upload bundle"
  timeout 900 colab upload -s "$NAME" "colab_bundle_$MODE.zip" "/content/colab_bundle_$MODE.zip" >>"$LOG" 2>&1 || die upload
  if [ -z "${NO_DRIVE:-}" ]; then
    mounted() { pyexec 'import os; print("drive", os.path.ismount("/content/drive"))' | grep -q "drive True"; }
    # interactive only: headless, its auth prompt blocks the kernel and the mount from another terminal queues behind it
    # --foreground: plain timeout puts drivemount in a background process group, so its Enter prompt stops on SIGTTIN
    [ -t 0 ] && timeout --foreground 600 colab drivemount -s "$NAME" 2>&1 | tee -a "$LOG" | grep -aE "colab\]|https:"
    # keep the loop's answer: a second probe can hit a dropped CLI connection right after a good mount (16 Oct a100v9)
    ok=0; for _ in $(seq 60); do mounted && { ok=1; break; }; say "waiting for Drive: run 'colab drivemount -s $NAME' in a terminal"; sleep 15; done
    [ $ok = 1 ] || die "Drive not mounted (or rerun with NO_DRIVE=1)"
  fi
  say "Part A (progress: /content/progress.json, Drive qwen35_waf/)"
  timeout $(( CELL_S + 600 )) colab exec -s "$NAME" "${ENV[@]}" --timeout "$CELL_S" -f partA.ipynb >>"$LOG" 2>&1
  mv -f partA_output.ipynb "$OUT/partA.ipynb" 2>/dev/null
fi
# the CLI connection can drop while the kernel trains on: wait for it instead of stopping the VM
until fetch part_a.json; do
  # one failed status poll must not stop a training VM (the EXIT trap stops it when KEEP=0)
  busy || { sleep 30; busy; } || { sleep 60; busy; } || die "Part A ended without part_a.json (see $OUT/partA.ipynb)"
  KEEP=1; fetch progress.json && say "training: $(cat "$OUT/progress.json")"; sleep 300; KEEP=0
done
fetch train_log.jsonl; fetch progress.json
say "Part A done: $(python3 -c "import json;a=json.load(open('$OUT/part_a.json'));print({k:a['train'][k] for k in ('global_steps','tokens_per_s_excl_warmup','peak_mem_gb','wall_s')}, a['predictions'])")"
[ -n "${PART_A_ONLY:-}" ] && { say "PART_A_ONLY: done"; exit 0; }

fresh() { pyexec 'import sys,subprocess; m=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip(); print("FRESH" if "torch" not in sys.modules and int(m) < 1000 else f"STALE {m} MiB")' | grep -q FRESH; }
for try in 1 2; do  # restart-kernel can time out silently
  say "restart kernel ($try)"; timeout 300 colab restart-kernel -s "$NAME" >>"$LOG" 2>&1; sleep 10
  fresh && break
  [ $try = 2 ] && die "kernel restart not verified (torch still loaded or GPU memory in use)"
done

say "Part B"
timeout $(( CELL_S + 600 )) colab exec -s "$NAME" "${ENV[@]}" --timeout "$CELL_S" -f partB.ipynb >>"$LOG" 2>&1
mv -f partB_output.ipynb "$OUT/partB.ipynb" 2>/dev/null
fetch "$REPORT" || die "Part B wrote no $REPORT (see $OUT/partB.ipynb)"
fetch "SCORES_$MODE.jsonl"
[ -z "$DRY" ] && fetch "export_$MODE.zip"
say "passed: $(python3 -c "import json;print(json.load(open('$OUT/$REPORT'))['passed'])") -> $OUT"

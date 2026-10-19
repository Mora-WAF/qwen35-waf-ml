#!/usr/bin/env bash
# One Gemma TPU run through the colab CLI: new v5e-1 session -> install -> upload data + trainer -> train + score as a
# detached process (a kernel that imports jax would hold the TPU) -> poll -> download -> stop.
# usage: training/gemma_tpu/run_tpu.sh NAME DATA_TGZ [ENV=VAL ...]
#   training/gemma_tpu/run_tpu.sh ep2 reports/gemma_tpu/colab/gemma_data.tgz GEMMA_EPOCHS=2
# Outputs: reports/gemma_tpu/colab_NAME/. Rerun the same line after a lost connection: it reuses the session and
# resumes from the VM's ckpt.npz (a lost runtime restarts from scratch).
set -uo pipefail
NAME=$1 TGZ=$2; shift 2
cd "$(dirname "$0")/../.."
OUT=reports/gemma_tpu/colab_$NAME; mkdir -p "$OUT"; LOG=$OUT/driver.log
say() { echo "[run_tpu $(date +%T)] $*" | tee -a "$LOG"; }
trap 'timeout 180 colab stop -s "$NAME" >>"$LOG" 2>&1; say "stopped $NAME"' EXIT
sh_() {  # run a shell command on the VM through a Python cell (json-quoted)
  local q; q=$(python3 -c 'import sys, json; print(json.dumps(sys.argv[1]))' "$1")
  timeout $(( ${2:-120} + 60 )) colab exec -s "$NAME" --timeout "${2:-120}" 2>&1 \
    <<<"import subprocess; print(subprocess.run($q, shell=True, capture_output=True, text=True).stdout[-3000:])"
}
if colab sessions 2>/dev/null | grep -q "^\[$NAME\]"; then say "reusing $NAME"; else
  say "new TPU session $NAME"; timeout 900 colab new -s "$NAME" --tpu v5e1 >>"$LOG" 2>&1 || { colab sessions | tee -a "$LOG"; say "FAILED: colab new"; exit 1; }
  say "install tunix + qwix"
  sh_ "pip install -q 'git+https://github.com/google/tunix@ba95987a5a3f1d3cf8a680238c1600a0db6453af' qwix==0.1.8 2>&1 | tail -2" 840 >>"$LOG"
  timeout 300 colab restart-kernel -s "$NAME" >>"$LOG" 2>&1; sleep 10
  timeout 900 colab upload -s "$NAME" "$TGZ" /content/data.tgz >>"$LOG" 2>&1 || { say "FAILED: upload data"; exit 1; }
  sh_ "mkdir -p /content/data /content/out && tar xzf /content/data.tgz -C /content/data && ls /content/data | wc -l" >>"$LOG"
fi
timeout 300 colab upload -s "$NAME" training/gemma_tpu/train_tunix.py /content/train_tunix.py >>"$LOG" 2>&1 || { say "FAILED: upload trainer"; exit 1; }
if ! sh_ "pgrep -f '[t]rain_tunix.py train' >/dev/null && echo RUNNING; grep -c FULL_EXIT /content/out/full.log 2>/dev/null" | grep -qE "RUNNING|^1"; then
  say "launch train ($*)"
  sh_ "cd /content && setsid nohup bash -c 'env $* GEMMA_CKPT_EVERY=500 python train_tunix.py train >> out/full.log 2>&1; echo FULL_EXIT \$? >> out/full.log' > /dev/null 2>&1 & sleep 2; echo launched" >>"$LOG"
fi
for _ in $(seq 400); do  # ~33 h cap
  st=$(sh_ "grep -E \"'step'|scored|Traceback|FULL_EXIT|resumed\" /content/out/full.log | tail -2 | cut -c1-110")
  say "$(echo "$st" | tail -1)"
  echo "$st" | grep -q "FULL_EXIT" && break
  sleep 300
done
echo "$st" | grep -q "FULL_EXIT 0" || say "run did not exit 0 (see full.log)"
for f in full.log train_log.jsonl train_report.json lora.npz SCORES_val.jsonl SCORES_test.jsonl SCORES_test_unseen.jsonl; do
  timeout 900 colab download -s "$NAME" "/content/out/$f" "$OUT/$f" >>"$LOG" 2>&1 || say "no $f"
done
say "done -> $OUT"

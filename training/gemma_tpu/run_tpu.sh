#!/usr/bin/env bash
# One Gemma TPU run through the colab CLI: new v5e-1 session -> install -> upload data + trainer -> (Drive mount) ->
# train + score as a detached process (a kernel that imports jax would hold the TPU) -> poll -> download -> stop.
# usage: [DRIVE=0] training/gemma_tpu/run_tpu.sh NAME DATA_TGZ [ENV=VAL ...]
#   training/gemma_tpu/run_tpu.sh cicfix reports/gemma_tpu/colab/gemma_data_cicfix.tgz
# Keep this machine on: the 5-min polls are the client activity a free-tier runtime needs (19 Oct: a run with no
# client attached was reclaimed within minutes). DRIVE=1 (default): wait up to 15 min for `colab drivemount -s NAME`
# in another terminal, then mirror /content/out to MyDrive/waf_gemma/colab_NAME every 5 min (ckpt.npz included).
# Outputs: reports/gemma_tpu/colab_NAME/. Rerun the same line after a lost connection or a lost VM: it reuses the
# session, or on a new VM copies the Drive mirror back so the trainer resumes from its ckpt.npz.
set -uo pipefail
NAME=$1 TGZ=$2; shift 2
DRIVE=${DRIVE:-1} DST=/content/drive/MyDrive/waf_gemma/colab_$NAME
cd "$(dirname "$0")/../.."
OUT=reports/gemma_tpu/colab_$NAME; mkdir -p "$OUT"; LOG=$OUT/driver.log
say() { echo "[run_tpu $(date +%T)] $*" | tee -a "$LOG"; }
trap 'timeout 180 colab stop -s "$NAME" >>"$LOG" 2>&1; say "stopped $NAME"' EXIT
sh_() {  # run a shell command on the VM through a Python cell (json-quoted)
  local q; q=$(python3 -c 'import sys, json; print(json.dumps(sys.argv[1]))' "$1")
  timeout $(( ${2:-120} + 60 )) colab exec -s "$NAME" --timeout "${2:-120}" 2>&1 \
    <<<"import subprocess; print(subprocess.run($q, shell=True, capture_output=True, text=True, errors='replace').stdout[-3000:])"
}
alive() { timeout 120 colab sessions 2>/dev/null | grep -q "^\[$NAME\]"; }
adopt() {  # `colab new` past the CLI's 120 s ReadTimeout can still assign a runtime, listed as [?]: take it over
  local ep; ep=$(timeout 120 colab sessions 2>/dev/null | sed -n 's/^\[?\] \(tpu-v5e1-[^ ]*\).*/\1/p' | head -1)
  [ -n "$ep" ] || return 1
  "$(head -1 "$(command -v colab)" | cut -c3-)" -c "from colab_cli.state import StateStore, SessionState as S
StateStore().add(S(name='$NAME', token='', url='', endpoint='$ep', accelerator='V5E1', variant='TPU'))" && say "adopted $ep"
}
if alive; then say "reusing $NAME"; else
  say "new TPU session $NAME"
  for try in 1 2 3; do
    timeout 900 colab new -s "$NAME" --tpu v5e1 >>"$LOG" 2>&1 && break
    sleep 20; alive || adopt && break
    [ $try = 3 ] && { timeout 120 colab sessions | tee -a "$LOG"; say "FAILED: colab new"; exit 1; }
    say "colab new failed (try $try), retry in 2 min"; sleep 120
  done
  say "install tunix + qwix"
  sh_ "pip install -q 'git+https://github.com/google/tunix@ba95987a5a3f1d3cf8a680238c1600a0db6453af' qwix==0.1.8 2>&1 | tail -2" 840 >>"$LOG"
  timeout 300 colab restart-kernel -s "$NAME" >>"$LOG" 2>&1; sleep 10
  timeout 900 colab upload -s "$NAME" "$TGZ" /content/data.tgz >>"$LOG" 2>&1 || { say "FAILED: upload data"; exit 1; }
  sh_ "mkdir -p /content/data /content/out && tar xzf /content/data.tgz -C /content/data && ls /content/data | wc -l" >>"$LOG"
fi
timeout 300 colab upload -s "$NAME" training/gemma_tpu/train_tunix.py /content/train_tunix.py >>"$LOG" 2>&1 || { say "FAILED: upload trainer"; exit 1; }
if [ "$DRIVE" = 1 ]; then
  if ! sh_ "mountpoint -q /content/drive && echo MOUNTED" | grep -q MOUNTED; then
    say "ACTION: run 'colab drivemount -s $NAME' in another terminal (waiting up to 15 min)"
    for _ in $(seq 30); do sh_ "mountpoint -q /content/drive && echo MOUNTED" | grep -q MOUNTED && break; sleep 30; done
  fi
  if sh_ "mountpoint -q /content/drive && echo MOUNTED" | grep -q MOUNTED; then
    say "Drive mirror -> $DST"
    # a new VM resumes from the mirror; the mirror loop runs until the VM goes
    sh_ "mkdir -p $DST /content/out; [ -f $DST/ckpt.npz ] && [ ! -f /content/out/ckpt.npz ] && rsync -a $DST/ /content/out/ && echo 'restored from Drive'; [ -f /content/.mirror ] || { touch /content/.mirror; setsid nohup bash -c 'while sleep 300; do rsync -a /content/out/ $DST/; done' >/dev/null 2>&1 & }; echo ok" >>"$LOG"
  else say "no Drive mount: continuing without the mirror (a lost VM restarts from scratch)"; DRIVE=0; fi
fi
if ! sh_ "pgrep -f '[t]rain_tunix.py train' >/dev/null && echo RUNNING; grep -c FULL_EXIT /content/out/full.log 2>/dev/null" | grep -qE "RUNNING|^1"; then
  say "launch train ($*)"
  sh_ "cd /content && setsid nohup bash -c 'env GEMMA_CKPT_EVERY=250 $* python train_tunix.py train >> out/full.log 2>&1; echo FULL_EXIT \$? >> out/full.log' > /dev/null 2>&1 & sleep 2; echo launched" >>"$LOG"
fi
for _ in $(seq 400); do  # ~33 h cap
  st=$(sh_ "grep -E \"'step'|scored|Traceback|FULL_EXIT|resumed\" /content/out/full.log | tail -2 | cut -c1-110")
  say "$(echo "$st" | tail -1)"
  echo "$st" | grep -q "FULL_EXIT" && break
  alive || { sleep 60; alive; } || { say "LOST: $NAME is gone from the server; rerun this line (DRIVE=1 resumes from the Drive ckpt)"; trap - EXIT; exit 1; }
  sleep 300
done
echo "$st" | grep -q "FULL_EXIT 0" || say "run did not exit 0 (see full.log)"
[ "$DRIVE" = 1 ] && sh_ "rsync -a /content/out/ $DST/ && echo synced" 600 >>"$LOG"
for f in full.log train_log.jsonl train_report.json lora.npz SCORES_val.jsonl SCORES_test.jsonl SCORES_test_unseen.jsonl; do
  timeout 900 colab download -s "$NAME" "/content/out/$f" "$OUT/$f" >>"$LOG" 2>&1 || say "no $f"
done
say "done -> $OUT"

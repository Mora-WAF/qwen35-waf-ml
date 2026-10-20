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
# Score only (no training), e.g. the public benchmarks for the ensemble (deploy/run_benchmarks.sh gemma):
#   GEMMA_SCORE="bench robustness" GEMMA_LORA=reports/gemma_tpu/colab_full/lora.npz \
#     training/gemma_tpu/run_tpu.sh bench reports/gemma_tpu/colab/gemma_data_bench.tgz
#   the adapter goes up in 8 MB parts (larger uploads fail), checksum-verified; no Drive mirror by default (DRIVE=0).
set -uo pipefail
NAME=$1 TGZ=$2; shift 2
cd "$(dirname "$0")/../.."  # paths (DATA_TGZ, GEMMA_LORA) are relative to the repository root
SCORE=${GEMMA_SCORE:-}
# GEMMA_SCORE with DRIVE=1 and no GEMMA_LORA: finish a lost run's scoring from the adapter in the Drive mirror
# (no adapter there or no mount: falls back to a full train run)
if [ -n "$SCORE" ]; then DRIVE=${DRIVE:-0}; [ -f "${GEMMA_LORA:-}" ] || [ "$DRIVE" = 1 ] || { echo "GEMMA_SCORE needs GEMMA_LORA=<lora.npz> or DRIVE=1"; exit 1; }
else DRIVE=${DRIVE:-1}; fi
DST=/content/drive/MyDrive/waf_gemma/colab_$NAME
OUT=reports/gemma_tpu/colab_$NAME; mkdir -p "$OUT"; LOG=$OUT/driver.log
say() { echo "[run_tpu $(date +%T)] $*" | tee -a "$LOG"; }
trap 'timeout 180 colab stop -s "$NAME" >>"$LOG" 2>&1; say "stopped $NAME"' EXIT
sh_() {  # run a shell command on the VM through a Python cell (json-quoted)
  local q; q=$(python3 -c 'import sys, json; print(json.dumps(sys.argv[1]))' "$1")
  timeout $(( ${2:-120} + 60 )) colab exec -s "$NAME" --timeout "${2:-120}" 2>&1 \
    <<<"import subprocess; print(subprocess.run($q, shell=True, capture_output=True, text=True, errors='replace').stdout[-3000:])"
}
alive() { timeout 120 colab sessions 2>/dev/null | grep -q "^\[$NAME\]"; }
up_parts() {  # up_parts LOCAL STEM: /content/STEM.part.* -> verified /content/STEM.<ext of LOCAL> (8 MB parts)
  local abs ext p f ok; abs=$(realpath "$1"); p=$(mktemp -d)
  case "$1" in *.npz) ext=npz;; *) ext=tgz;; esac
  (cd "$p" && split -b 8m -d "$abs" "$2.part." && sha256sum "$abs" | cut -c1-64 > "$2.sha256") || return 1
  for f in "$p"/"$2".part.* "$p/$2.sha256"; do
    ok=0; for _ in 1 2 3; do timeout 900 colab upload -s "$NAME" "$f" "/content/$(basename "$f")" >>"$LOG" 2>&1 && { ok=1; break; }; sleep 20; done
    [ $ok = 1 ] || { rm -rf "$p"; return 1; }
  done
  rm -rf "$p"
  sh_ "cd /content && cat $2.part.* > $2.$ext && sha256sum $2.$ext | cut -c1-64 | cmp -s - $2.sha256 && echo PARTS-OK" | grep -q PARTS-OK
}
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
  up_parts "$TGZ" data || { say "FAILED: upload data"; exit 1; }  # parts: uploads above ~10 MB fail
  sh_ "mkdir -p /content/data /content/out && tar xzf /content/data.tgz -C /content/data && ls /content/data | wc -l" >>"$LOG"
fi
timeout 300 colab upload -s "$NAME" training/gemma_tpu/train_tunix.py /content/train_tunix.py >>"$LOG" 2>&1 || { say "FAILED: upload trainer"; exit 1; }
if [ -n "$SCORE" ] && [ -n "${GEMMA_LORA:-}" ] && ! sh_ "[ -f /content/out/lora.npz ] && sha256sum /content/out/lora.npz | cut -c1-64 | cmp -s - /content/lora.sha256 && echo LORA-OK" | grep -q LORA-OK; then
  say "upload adapter $GEMMA_LORA in parts"
  up_parts "$GEMMA_LORA" lora && sh_ "mkdir -p /content/out && cp /content/lora.npz /content/out/lora.npz && echo copied" | grep -q copied \
    || { say "FAILED: adapter upload / checksum"; exit 1; }
fi
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
if [ -n "$SCORE" ] && [ -z "${GEMMA_LORA:-}" ] && ! sh_ "[ -f /content/out/lora.npz ] && echo HAVE" | grep -q HAVE; then
  say "no adapter in the Drive mirror: full train run instead"; SCORE=""
fi
CMD=${SCORE:+score $SCORE}; CMD=${CMD:-train}
if ! sh_ "pgrep -f '[t]rain_tunix.py' >/dev/null && echo RUNNING; grep -c FULL_EXIT /content/out/full.log 2>/dev/null" | grep -qE "RUNNING|^1"; then
  say "launch $CMD ($*)"
  sh_ "cd /content && setsid nohup bash -c 'env GEMMA_CKPT_EVERY=250 $* python train_tunix.py $CMD >> out/full.log 2>&1; echo FULL_EXIT \$? >> out/full.log' > /dev/null 2>&1 & sleep 2; echo launched" >>"$LOG"
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
FILES="full.log train_log.jsonl train_report.json lora.npz SCORES_val.jsonl SCORES_test.jsonl SCORES_test_unseen.jsonl"
[ -n "$SCORE" ] && [ -n "${GEMMA_LORA:-}" ] && FILES="full.log $(for s in $SCORE; do printf 'SCORES_%s.jsonl ' "$s"; done)"
for f in $FILES; do
  timeout 900 colab download -s "$NAME" "/content/out/$f" "$OUT/$f" >>"$LOG" 2>&1 || say "no $f"
done
say "done -> $OUT"

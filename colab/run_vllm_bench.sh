#!/usr/bin/env bash
# vLLM vs HF transformers benchmark of the classifier (v10 by default) on a Colab GPU (no Drive, no browser step).
#   colab/run_vllm_bench.sh [NAME] [GPU] [N]     defaults: vllmbench L4 400
# new VM -> upload model package + N test records with their A100 scores -> pip -> merge -> hf -> vllm -> report
# -> download reports/vllm_bench_<GPU>.json -> stop the VM (always, via the EXIT trap).
# Env: RUN_DIR BUNDLE_ZIP (deploy.sh args, e.g. reports/colab_a100v11 colab/colab_bundle_fast_v11.zip), FP_IN FP_OUT, STAGES.
#      GEMMA_LORA GEMMA_POLICY GEMMA_SELFCHECK GEMMA_OUT: the Gemma adapter (8 MB parts), its frozen policy and the TPU
#      selfcheck records for the gemma_fp stage (same records as fp, Gemma on HF transformers); GEMMA_ROBUST
#      GEMMA_ROBUST_OUT: robustness probes ({id, split, kind, label, telemetry}) -> Gemma's raw class scores.
# Public benchmarks with the v11 model (all three in one VM): deploy/run_benchmarks.sh qwen
set -uo pipefail
NAME=${1:-vllmbench} GPU=${2:-L4} N=${3:-400}
cd "$(dirname "$0")/.."
OUT=reports/colab_$NAME; mkdir -p "$OUT"; LOG="$OUT/driver.log"
say() { echo "[vllm_bench $(date +%T)] $*" | tee -a "$LOG"; }
die() { say "FAILED: $*"; exit 1; }
trap 'timeout 180 colab stop -s "$NAME" >>"$LOG" 2>&1; colab sessions 2>&1 | tail -3 | tee -a "$LOG"' EXIT
step() {  # step SECONDS PYTHON-CODE: run in the kernel, output to the log; fail when the marker is missing
  local t=$1 code=$2 n=$RANDOM$RANDOM
  timeout $((t + 120)) colab exec -s "$NAME" --timeout "$t" 2>&1 <<<"$code
print('STEP-OK $n')" | tee -a "$LOG" | grep -q "STEP-OK $n"
}

S=$(mktemp -d); trap 'rm -rf "$S"; timeout 180 colab stop -s "$NAME" >>"$LOG" 2>&1; colab sessions 2>&1 | tail -3 | tee -a "$LOG"' EXIT
PACKAGE_ONLY="$S/model.tar.gz" deploy/deploy.sh ${RUN_DIR:-} ${BUNDLE_ZIP:-} >>"$LOG" 2>&1 || die "package"
mkdir "$S/pkg" && tar -xzf "$S/model.tar.gz" -C "$S/pkg" && (cd "$S/pkg" && zip -qr "$S/pkg.zip" .) || die "zip"  # .tar.gz uploads fail with HTTP 500
.venv/bin/python - "$S/sample.jsonl" "$N" "${RUN_DIR:-reports/colab_a100v10}" "${BUNDLE_ZIP:-colab/colab_bundle_fast_v10.zip}" <<'EOF'
import json, random, sys
from training.prepare import deserialize_telemetry
f = open(f"{sys.argv[3]}/SCORES_fast.jsonl"); f.readline()
scores = {r["id"]: r["scores"] for r in map(json.loads, f) if r["split"] == "test"}
import zipfile  # the run's own test set (its A100 scores belong to it), not whatever datasets_ready/ holds now
rows = [json.loads(l) for l in zipfile.ZipFile(sys.argv[4]).open("datasets_ready/fast_test.jsonl")]
rows = random.Random(7).sample([r for r in rows if r["id"] in scores], int(sys.argv[2]))
open(sys.argv[1], "w").write("".join(json.dumps({"id": r["id"], "telemetry": deserialize_telemetry(r["messages"][1]["content"]),
                                                 "scores": scores[r["id"]]}) + "\n" for r in rows))
EOF
say "new $GPU session $NAME"
timeout 900 colab new -s "$NAME" --gpu "$GPU" >>"$LOG" 2>&1 || die "colab new"
ok=0; for _ in 1 2 3; do  # the first call after READY can lose the connection (18 Oct)
  step 300 "import subprocess; print(subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True).stdout)" && { ok=1; break; }; sleep 30
done; [ $ok = 1 ] || die "kernel not answering"
up() { for _ in 1 2 3; do timeout 900 colab upload -s "$NAME" "$1" "$2" >>"$LOG" 2>&1 && return 0; sleep 20; done; return 1; }  # 500s right after READY
# one Jupyter contents request carries ~10 MB at most (the 79 MB package fails with HTTP 500): upload 8 MB parts
(cd "$S" && split -b 8m -d pkg.zip pkg.part. && sha256sum pkg.zip | cut -c1-64 > pkg.sha256)
for f in "$S"/pkg.part.* "$S/pkg.sha256" "$S/sample.jsonl"; do up "$f" "/content/$(basename "$f")" || die "upload $(basename "$f")"; done
up colab/vllm_bench.py /content/vllm_bench.py || die "upload script"
if [ -n "${GEMMA_LORA:-}" ]; then  # the TPU adapter (95 MB) in 8 MB parts, checksum-verified on the VM
  GL_ABS=$(realpath "$GEMMA_LORA") && mkdir "$S/gl" && (cd "$S/gl" && split -b 8m -d "$GL_ABS" gl.part. && sha256sum "$GL_ABS" | cut -c1-64 > gl.sha256) || die "split gemma adapter"
  for f in "$S"/gl/gl.part.* "$S/gl/gl.sha256"; do up "$f" "/content/$(basename "$f")" || die "upload $(basename "$f")"; done
  up "${GEMMA_POLICY:-config/policy_gemma.yaml}" /content/policy_gemma.yaml || die "upload gemma policy"
  up "$GEMMA_SELFCHECK" /content/gemma_selfcheck.jsonl || die "upload gemma selfcheck"
  if [ -n "${GEMMA_ROBUST:-}" ]; then up "$GEMMA_ROBUST" /content/robust_in.jsonl || die "upload robustness set"; fi
fi
if [ -n "${FP_IN:-}" ]; then  # gzip; in 8 MB parts like the package (every benchmark together is larger)
  FP_ABS=$(realpath "$FP_IN") && mkdir "$S/fp" && (cd "$S/fp" && split -b 8m -d "$FP_ABS" fp.part. && sha256sum "$FP_ABS" | cut -c1-64 > fp.sha256) || die "split fp input"
  for f in "$S"/fp/fp.part.* "$S/fp/fp.sha256"; do up "$f" "/content/$(basename "$f")" || die "upload $(basename "$f")"; done
fi
# a long call can lose the CLI connection (01:58: "Connection was lost" during merge): every stage runs detached on
# the VM and is polled with short calls, so a dropped connection costs one poll, not the stage
RUNS=0
run() {  # run SECONDS "shell command"
  local t=$1 cmd=$2 id=stage$((++RUNS)) rc="" waited=0 out
  local started=0
  for _ in 1 2 3 4 5; do  # the start call can lose the connection too (02:26); never launch a stage twice
    step 120 "import os, subprocess
if not os.path.exists('/content/$id.log'):
    subprocess.Popen(['bash', '-c', r'''($cmd) > /content/$id.log 2>&1; echo \$? > /content/$id.rc'''], start_new_session=True)" && { started=1; break; }
    say "start of $id not confirmed, retrying"; sleep 20
  done
  [ $started = 1 ] || return 1
  while [ $waited -lt "$t" ]; do
    sleep 30; waited=$((waited + 30))
    out=$(timeout 150 colab exec -s "$NAME" --timeout 60 2>&1 <<<"import os
print('RC=' + (open('/content/$id.rc').read().strip() if os.path.exists('/content/$id.rc') else 'RUNNING'))
print(open('/content/$id.log', errors='replace').read()[-1500:] if os.path.exists('/content/$id.log') else '')")
    rc=$(grep -ao "RC=[A-Z0-9]*" <<<"$out" | head -1 | cut -d= -f2)
    case "$rc" in RUNNING|"") continue;; 0) echo "$out" | tail -8 >>"$LOG"; return 0;; *) echo "$out" >>"$LOG"; return 1;; esac
  done
  say "timeout after ${t}s"; return 1
}
say "unpack + pip (vllm 0.30.0 brings torch 2.13)"
run 600 "cd /content && cat pkg.part.* > pkg.zip && sha256sum pkg.zip | cut -c1-64 | cmp -s - pkg.sha256 && mkdir -p pkg && cd pkg && unzip -q -o ../pkg.zip" || die "unpack (checksum)"
if [ -n "${FP_IN:-}" ]; then
  run 300 "cd /content && cat fp.part.* > fp_in.jsonl.gz && sha256sum fp_in.jsonl.gz | cut -c1-64 | cmp -s - fp.sha256" || die "fp input (checksum)"
fi
if [ -n "${GEMMA_LORA:-}" ]; then
  run 300 "cd /content && cat gl.part.* > gemma_lora.npz && sha256sum gemma_lora.npz | cut -c1-64 | cmp -s - gl.sha256" || die "gemma adapter (checksum)"
fi
run 1500 "pip uninstall -y -q torchao; pip install -q vllm==0.30.0 transformers==5.17.0 peft==0.21.2 jsonschema" || die pip  # Colab torchao 0.10 breaks peft imports
for st in ${STAGES:-merge:2400 hf:2400 vllm:2400 report:300}; do  # STAGES="merge:2400 vllm_gen:2400 report:300"
  say "stage ${st%%:*}"; run "${st##*:}" "cd /content && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 vllm_bench.py ${st%%:*}" || die "stage ${st%%:*} (see $LOG)"
done
if [ -n "${FP_IN:-}" ]; then
  timeout 600 colab download -s "$NAME" /content/fp_out.jsonl "${FP_OUT:-reports/benchmarks/openappsec_legit.results.jsonl}" >>"$LOG" 2>&1 || die "download fp"
  say "done -> ${FP_OUT:-reports/benchmarks/openappsec_legit.results.jsonl}"
fi
case " ${STAGES:-} " in *gemma_fp*)
  timeout 600 colab download -s "$NAME" /content/gemma_fp_out.jsonl "${GEMMA_OUT:-reports/benchmarks/bench_gemma.fp_out.jsonl}" >>"$LOG" 2>&1 || die "download gemma fp"
  if [ -n "${GEMMA_ROBUST:-}" ]; then
    timeout 600 colab download -s "$NAME" /content/gemma_robustness_scores.jsonl "${GEMMA_ROBUST_OUT:-reports/gemma_tpu/SCORES_robustness_hf.jsonl}" >>"$LOG" 2>&1 || die "download gemma robustness"
  fi
  say "done -> ${GEMMA_OUT:-reports/benchmarks/bench_gemma.fp_out.jsonl}";;
esac
case " ${STAGES:-report} " in *report*)
  timeout 300 colab download -s "$NAME" /content/vllm_bench.json "reports/vllm_bench_$GPU.json" >>"$LOG" 2>&1 || die download
  say "done -> reports/vllm_bench_$GPU.json"; cat "reports/vllm_bench_$GPU.json";;
esac

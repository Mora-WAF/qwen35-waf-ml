"""Package everything the Colab notebook needs into one zip (spec §9: no dependency on the
developer checkout, no paid generation rerun). Refuses to package a build whose gate failed.

usage: python -m training.make_bundle [--prefix pilot_] [--allow-failed-gate]
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ["training/__init__.py", "training/prepare.py", "training/colab_helpers.py", "training/tpu_train.py", "eval/__init__.py", "eval/policy.py",
        "config/taxonomy.yaml", "config/policy.yaml",
        "schemas/classification.schema.json", "schemas/decision.schema.json",
        "schemas/record.schema.json", "schemas/telemetry.schema.json",
        "reports/pre_colab_gate.json", "datasets_ready/manifest.json"]


def main(prefix, allow_failed):
    gate = json.load(open(ROOT / "reports/pre_colab_gate.json"))
    if gate["prefix"] != prefix:
        raise SystemExit(f"gate report is for prefix {gate['prefix']!r}, rerun validate_pre_colab --prefix {prefix}")
    if not gate["passed"] and not allow_failed:
        raise SystemExit(f"pre-Colab gate failed ({gate['n_fail']} failures); refusing to bundle")
    manifest = json.load(open(ROOT / "datasets_ready/manifest.json"))
    data = [f"datasets_ready/{f}" for f in manifest["files"] if f.startswith(prefix)]
    out = ROOT / "colab" / f"colab_bundle_{prefix.rstrip('_') or 'full'}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in CODE + data:
            z.write(ROOT / rel, rel)
        z.writestr("BUNDLE_INFO.json", json.dumps(
            {"prefix": prefix, "gate_passed": gate["passed"], "files": CODE + data}, indent=1))
    print(out, hashlib.sha256(out.read_bytes()).hexdigest(), f"{out.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pilot_")
    ap.add_argument("--allow-failed-gate", action="store_true")
    a = ap.parse_args()
    main(a.prefix, a.allow_failed_gate)

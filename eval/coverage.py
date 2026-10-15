"""Coverage matrix (spec §7/§8): per class, how many scenario families, delivery locations, encoding
layers, sources and flow variants the build exercises. Thin rows are where new families are needed.

usage: python -m eval.coverage [--prefix pilot_]   -> reports/coverage.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_FAMILIES = 6   # below this a class is reported as thin (the pilot target; the full build needs more)
MIN_SEED_GROUPS = 30


def main(prefix):
    rows = defaultdict(lambda: defaultdict(Counter))
    n = Counter()
    for fn in sorted((ROOT / "datasets_ready").glob(f"{prefix}*.jsonl")):
        for line in open(fn):
            m = json.loads(line)["metadata"]
            c = m["threat_type"]
            n[c] += 1
            rows[c]["provenance"][m["provenance"]] += 1
            rows[c]["template_family"][m["template_family"]] += 1
            rows[c]["lineage"][m["lineage_group"]] += 1
            for t in m["mechanism_tags"]:
                kind, _, val = t.partition(":")
                rows[c][kind][val] += 1
    rep = {}
    for c in sorted(n):
        r = rows[c]
        fams = len(r["template_family"]) + len(r["family"]) + len(r["variant"]) + len(r["src"])
        rep[c] = {"records": n[c], "family_like_units": fams, "lineage_groups": len(r["lineage"]),
                  **{k: dict(v.most_common()) for k, v in r.items() if k != "lineage"}}
    # seeded classes get their diversity from independent seeds (lineage groups), synthetic ones from families
    thin = sorted(c for c, v in rep.items() if v["family_like_units"] < MIN_FAMILIES and v["lineage_groups"] < MIN_SEED_GROUPS)
    out = {"prefix": prefix, "min_families": MIN_FAMILIES, "thin_classes": thin, "classes": rep}
    (ROOT / "reports/coverage.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"thin_classes": thin,
                      "family_like_units": {c: v["family_like_units"] for c, v in rep.items()},
                      "locations": {c: sorted(v.get("loc", {})) for c, v in rep.items()}}, indent=0)[:3000])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pilot_")
    main(ap.parse_args().prefix)

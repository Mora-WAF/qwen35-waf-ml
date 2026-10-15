"""Raw cache -> normalized observed seeds (spec §3, §4).

A seed = one attack string (or one raw HTTP request block) with provenance back to an
immutable source revision and file. Seeds carry NO label evidence of exploitability; they
are material for the generator (§6), which produces `hybrid` telemetry from them.

Deterministic. Writes JSONL to data/seeds/<class>.jsonl + a per-run stats dict.
"""
import hashlib
import json
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT.parent / "raw_cache"
OUT = ROOT / "data" / "seeds"


def seed_id(source, cls, text):
    return f"{source}:{cls}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"


def parse_lines(text):
    """One payload per non-blank line; drop comments and obvious section markers."""
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//") or set(s) <= set("-=#"):
            continue
        yield s


def parse_http_blocks(text):
    """Blank-line-separated raw HTTP request blocks. Framing is preserved verbatim; a block
    is a smuggling *fixture*, not proof of a successful desync (§5)."""
    block = []
    for ln in text.split("\n"):
        if ln.strip() == "" and block and block[-1].strip() == "":
            b = "\n".join(block).strip("\n")
            if b:
                yield b
            block = []
        else:
            block.append(ln)
    tail = "\n".join(block).strip("\n")
    if tail:
        yield tail


MD_LINE_LANGS = {"sql", "ps1", "powershell", ""}   # PATT lists payloads in these fences
MD_BLOCK_LANGS = {"json"}                           # one payload per block
# python/ruby/bash/js fences are tooling scripts, not request payloads -> skipped (spec §3)


def parse_md_codeblocks(text):
    import re
    for lang, body in re.findall(r"```([A-Za-z0-9]*)\n(.*?)```", text, re.S):
        lang = lang.lower()
        if lang in MD_BLOCK_LANGS:
            b = " ".join(body.split())
            if b:
                yield b
        elif lang in MD_LINE_LANGS:
            yield from parse_lines(body)


PARSERS = {"lines": parse_lines, "http_blocks": parse_http_blocks, "md_codeblocks": parse_md_codeblocks}

# ---- seed quality (spec §6: quarantine unsupported labels) ----------------------------------
import re  # noqa: E402

METACHAR = re.compile(r"[^A-Za-z0-9 _.\-]")
PROSE = re.compile(r"\b(the|this|that|payload|example|used|will|which|with)\b", re.I)


def quality_reject(payload, parser):
    """Reason string if the seed is not usable evidence for its class, else None."""
    if parser == "http_blocks":
        return None if "\n" in payload and " HTTP/" in payload.split("\n", 1)[0] else "not_a_request"
    if len(payload) <= 3:
        return "too_short_ambiguous"       # '*', ')(' , "'" : benign in most contexts
    if not METACHAR.search(payload):
        return "no_attack_syntax"          # bare words ('onload', 'union select') are hard-negative territory
    if len(payload.split()) > 6 and len(PROSE.findall(payload)) >= 2:
        return "prose_not_payload"
    return None


def deobfuscate(payload):
    """View used only for class-syntax checks: url-decoded (2 rounds), inline comments removed."""
    from urllib.parse import unquote
    return re.sub(r"/\*.*?\*/", "", unquote(unquote(payload)))


OBFUSCATION = re.compile(
    r"%[0-9a-f]{2}|\\x[0-9a-f]{2}|\\u00[0-9a-f]{2}|\b0x[0-9a-f]{6,}"   # an encoding layer inside the payload itself
    r"|\b(?:n?char|chr)\s*\(\s*\d+"                                     # char-code string building
    r"|\w/\*.*?\*/\w|/\*!\d*",                                           # keyword-splitting / versioned comments
    re.I)
MIXED_CASE_KW = re.compile(r"\b(select|union|where|sleep|waitfor|insert|from)\b", re.I)


def sqli_seed_plain(payload):
    from urllib.parse import unquote
    return unquote(payload) if re.search(r"%[0-9a-f]{2}", payload, re.I) else payload


def is_obfuscated_sql(payload):
    """Taxonomy: sqli_obfuscated wins when an encoding or obfuscation layer is needed to form the injection.
    Applied after sqli_seed_plain to seeds a list files under classic SQLi (FuzzDB/SecLists mix both); a %XX
    left after one decode is a nested encoding layer."""
    return bool(OBFUSCATION.search(payload)) or any(
        not (w.islower() or w.isupper() or w.istitle()) for w in MIXED_CASE_KW.findall(payload))


def canon(payload):
    """Near-duplicate key: case, whitespace and digit runs folded."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", payload.strip().lower()))


def sources_index(root=ROOT):
    return {s["id"]: s for s in yaml.safe_load(open(root / "config/sources.yaml"))["sources"]}


def rules(root=ROOT):
    return yaml.safe_load(open(root / "config/seed_map.yaml"))["rules"]


REJECTED = []  # quality rejects, kept for the robustness set (insufficient-context cases)


def iter_seeds(root=ROOT, rejects=None):
    srcs, seen = sources_index(root), set()
    rejects = rejects if rejects is not None else Counter()
    for rule in rules(root):
        src = srcs[rule["source"]]
        if not src["enabled"]:
            continue
        base = RAW / src["id"] / src["revision"]
        parse = PARSERS[rule["parser"]]
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name == "MANIFEST.json":
                continue
            rel = str(path.relative_to(base))
            if not fnmatch(rel, rule["glob"]):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            req = re.compile(rule["require"]) if rule.get("require") else None
            for payload in parse(text):
                why = quality_reject(payload, rule["parser"])
                if not why and req and not req.search(deobfuscate(payload)):
                    why = "fails_class_syntax"
                if why:
                    rejects[(rule["class"], why)] += 1
                    REJECTED.append({"class": rule["class"], "reason": why, "payload": payload,
                                     "source": src["id"], "source_file": rel})
                    continue
                cls = rule["class"]
                if cls == "sqli_classic":
                    # list payloads come pre-URL-encoded for a query string; the generator re-encodes per location,
                    # so decode once and route on the layers that remain (re-check: raw %20 in a URL read as classic)
                    payload = sqli_seed_plain(payload)
                    if is_obfuscated_sql(payload):
                        cls = "sqli_obfuscated"
                sid = seed_id(src["id"], cls, payload)
                if sid in seen:  # exact-dup across files/rules
                    continue
                seen.add(sid)
                yield {"id": sid, "class": cls, "payload": payload,
                       "provenance": "observed", "parser": rule["parser"],
                       "source": src["id"], "source_revision": src["revision"],
                       "source_file": rel, "license_ref": src["license_spdx"],
                       "cluster": f"{cls}:" + hashlib.sha256(canon(payload).encode()).hexdigest()[:16]}


def build(root=ROOT):
    OUT.mkdir(parents=True, exist_ok=True)
    buckets, files, rejects = {}, {}, Counter()
    seeds_all = list(iter_seeds(root, rejects))
    # one primary class per payload (spec §4): resolve cross-class clusters by taxonomy precedence,
    # quarantine the rest as unresolved multi-label
    tax = yaml.safe_load(open(root / "config/taxonomy.yaml"))["classes"]
    by_canon = {}
    for sd in seeds_all:
        by_canon.setdefault(canon(sd["payload"]), set()).add(sd["class"])
    for sd in seeds_all:
        classes = by_canon[canon(sd["payload"])]
        if len(classes) > 1:
            winners = [c for c in classes if set(tax[c]["precedence"]["over"]) >= classes - {c}]
            if winners != [sd["class"]]:
                rejects[(sd["class"], "lost_precedence" if winners else "multi_class_unresolved")] += 1
                continue
        buckets.setdefault(sd["class"], []).append(sd)
    for cls, seeds in sorted(buckets.items()):
        p = OUT / f"{cls}.jsonl"
        with open(p, "w") as f:
            for s in seeds:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        files[cls] = {"seeds": len(seeds), "clusters": len({x["cluster"] for x in seeds})}
    rej = {}
    for (cls, why), n in rejects.items():
        rej.setdefault(cls, {})[why] = n
    with open(OUT / "rejected.jsonl", "w") as fh:
        for x in REJECTED:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    (OUT / "seed_stats.json").write_text(json.dumps({"accepted": files, "rejected": rej}, indent=1, sort_keys=True))
    return files


if __name__ == "__main__":
    for cls, n in sorted(build().items()):
        print(f"{cls}: {n}")
    print("rejected:", json.load(open(OUT / "seed_stats.json"))["rejected"])

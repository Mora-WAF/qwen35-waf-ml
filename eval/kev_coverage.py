"""Which known-exploited vulnerabilities (CISA KEV) can this classifier cover at all? Metadata only (CVE id, vendor,
product, CWE); no payloads.

Each KEV entry's CWEs map to one of:
  covered        a taxonomy class detects the technique from the request itself
  context        a class exists, but the verdict needs gateway context (authz, verifier, session) the sensor must supply
  not_detectable a request classifier cannot see it in principle (memory corruption, local privilege paths, supply
                 chain): network/endpoint controls, patching
  unmapped       generic or missing CWE (CWE-20 input validation, CWE-74 generic injection, none)
An entry counts as the best level among its CWEs. Output: reports/kev_coverage.json (+ per-class demand ranking).

usage: python -m eval.kev_coverage [KEV_JSON]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEV = ROOT.parent / "raw_cache" / "cisa_kev" / "known_exploited_vulnerabilities.json"
COVERED = {
    "rce_cmd_injection": ["CWE-77", "CWE-78", "CWE-88"],
    "path_traversal_lfi": ["CWE-22", "CWE-23", "CWE-35", "CWE-36", "CWE-73", "CWE-98"],
    "ssti": ["CWE-94", "CWE-95", "CWE-917", "CWE-1336"],  # code / expression-language injection
    "deserialization": ["CWE-502"],
    "xss_reflected_stored": ["CWE-79", "CWE-80"],
    "sqli_classic": ["CWE-89", "CWE-564"],
    "malicious_file_upload_webshell": ["CWE-434"],
    "ssrf_cloud_metadata": ["CWE-918"],
    "xxe": ["CWE-611", "CWE-776"],
    "open_redirect_cors": ["CWE-601", "CWE-942"],
    "crlf_header_injection": ["CWE-93", "CWE-113"],
    "ldap_xpath_injection": ["CWE-90", "CWE-643"],
    "nosqli": ["CWE-943"],
    "http_request_smuggling": ["CWE-444"],
    "csv_formula_injection": ["CWE-1236"],
    "prototype_pollution": ["CWE-1321"],
}
CONTEXT = {  # the class exists; the evidence is in gateway context, not the payload
    "jwt_auth_bypass / bfla_admin_function_probing": ["CWE-287", "CWE-288", "CWE-290", "CWE-294", "CWE-306",
                                                      "CWE-345", "CWE-347", "CWE-798", "CWE-521", "CWE-522"],
    "bola_idor_enumeration / bfla / bopla": ["CWE-284", "CWE-285", "CWE-639", "CWE-862", "CWE-863", "CWE-269",
                                             "CWE-264", "CWE-266", "CWE-915"],
    "race_condition_toctou_burst": ["CWE-362", "CWE-367"],
}
NOT_DETECTABLE = ["CWE-787", "CWE-416", "CWE-119", "CWE-122", "CWE-125", "CWE-190", "CWE-121", "CWE-120", "CWE-843",
                  "CWE-476", "CWE-415", "CWE-401", "CWE-763", "CWE-824", "CWE-191", "CWE-704", "CWE-908", "CWE-680",
                  "CWE-59", "CWE-506", "CWE-427", "CWE-426", "CWE-428", "CWE-668", "CWE-732", "CWE-276", "CWE-1188",
                  "CWE-693", "CWE-494", "CWE-829", "CWE-427", "CWE-114", "CWE-122", "CWE-129", "CWE-131", "CWE-134",
                  "CWE-252", "CWE-253", "CWE-367", "CWE-400", "CWE-404", "CWE-459", "CWE-665", "CWE-674", "CWE-681",
                  "CWE-682", "CWE-772", "CWE-822", "CWE-823", "CWE-825", "CWE-835"]
RANK = {"covered": 3, "context": 2, "not_detectable": 1, "unmapped": 0}


def level(cwe):
    for cls, cwes in COVERED.items():
        if cwe in cwes:
            return "covered", cls
    for cls, cwes in CONTEXT.items():
        if cwe in cwes:
            return "context", cls
    if cwe in NOT_DETECTABLE:
        return "not_detectable", None
    return "unmapped", None


def main(path=KEV):
    vulns = json.load(open(path))["vulnerabilities"]
    lv, demand, ransomware, unmapped_cwes = Counter(), Counter(), Counter(), Counter()
    for v in vulns:
        best = max((level(c) for c in (v.get("cwes") or [])), key=lambda x: RANK[x[0]], default=("unmapped", None))
        lv[best[0]] += 1
        if best[1]:
            demand[best[1]] += 1
            ransomware[best[1]] += v.get("knownRansomwareCampaignUse") == "Known"
        if best[0] == "unmapped":
            unmapped_cwes.update(v.get("cwes") or ["(none)"])
    n = len(vulns)
    rep = {"kev_entries": n, "catalog": Path(path).name,
           "coverage": {k: {"n": lv[k], "share": round(lv[k] / n, 3)} for k in RANK},
           "demand_by_class": [{"class": c, "kev_cves": k, "ransomware_known": ransomware[c]} for c, k in demand.most_common()],
           "top_unmapped_cwes": unmapped_cwes.most_common(10),
           "note": "metadata only; 'covered' means a class exists for the technique, not that every exploit is detected"}
    (ROOT / "reports/kev_coverage.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])

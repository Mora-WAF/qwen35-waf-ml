import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml  # noqa: E402
from eval.policy import decide, load, outside_envelope, validate_decision  # noqa: E402

TAX, POL, SCHEMA = load()
CAL = {**POL, "threshold": 0.9}  # pretend-fitted policy for logic tests
SHADOW = {**POL, "threshold": None}  # unfitted policy (the shipped file is frozen by eval.calibration --write-policy)


def run(raw, mode="L7_HTTP", ctx=None, conf=0.95, pol=CAL):
    d, diag = decide(raw, mode, ctx, conf, TAX, pol)
    validate_decision(d, TAX, SCHEMA)
    return d, diag


def out(c):
    return json.dumps({"threat_type": c})


def test_schema_enum_matches_taxonomy():
    cls = json.load(open(Path(__file__).parent.parent / "schemas/classification.schema.json"))
    assert cls["properties"]["threat_type"]["enum"] == list(TAX["classes"])


@pytest.mark.parametrize("raw", ["", "nope", '{"threat_type":"sqli"}', '["sqli_classic"]',
                                 '{"threat_type":"sqli_classic","x":1}', None,
                                 '{"threat_type":"sqli_classic"}<|im_start|>'])
def test_invalid_output_abstains(raw):
    d, diag = run(raw)
    assert d["verdict"] == "ABSTAIN" and diag["reason"] == "malformed_output"


def test_uncalibrated_is_shadow():
    d, diag = run(out("sqli_classic"), pol=SHADOW)
    assert d["verdict"] == "ABSTAIN" and d["confidence"] is None
    assert diag["tentative_class"] == "sqli_classic"
    assert run(out("sqli_classic"), conf=None)[0]["verdict"] == "ABSTAIN"
    assert run(out("sqli_classic"), conf=float("nan"))[0]["verdict"] == "ABSTAIN"


def test_block_allow_low_conf():
    d, _ = run(out("sqli_classic"))
    assert (d["verdict"], d["action"], d["domain"], d["rule_hint"]) == \
        ("BLOCK", "drop_request", "L7_INJECTION", "ml_sqli_classic")
    d, _ = run(out("benign_high_rate_legit"))
    assert (d["verdict"], d["action"], d["rule_hint"]) == ("ALLOW", "pass", "none")
    assert run(out("sqli_classic"), conf=0.5)[1]["reason"] == "low_confidence"


def test_mode_and_evidence_gates():
    assert run(out("syn_flood_ddos"), mode="L7_HTTP")[1]["reason"] == "unsupported_mode"
    assert run(out("bola_idor_enumeration"))[1]["reason"] == "missing_evidence"
    d, _ = run(out("bola_idor_enumeration"), ctx={"authz_context": {"subject": "u1"}})
    assert d["verdict"] == "BLOCK"


def test_alt_action_requires_eligibility():
    pol = {**CAL, "class_actions": {"credential_stuffing_bruteforce":
                                    {"verdict": "RATE_LIMIT", "eligibility": "rate_limit_eligible"}}}
    c = out("credential_stuffing_bruteforce")
    assert run(c, mode="L4_FLOW", pol=pol)[1]["reason"] == "ineligible_action"
    d, _ = run(c, mode="L4_FLOW", ctx={"rate_limit_eligible": True}, pol=pol)
    assert (d["verdict"], d["action"]) == ("RATE_LIMIT", "rate_limit_ip")


def test_flow_envelope_abstains_on_background_class_only():
    env = {**CAL, "flow_envelope": {"pkts_per_s": [1.0, 500.0]}}
    flood = {"pkts_per_s": 90000.0, "dst_port": 80}
    d, diag = decide(out("benign_standard_traffic"), "L4_FLOW", None, 0.99, TAX, env, flood)
    assert d["verdict"] == "ABSTAIN" and diag["reason"] == "out_of_envelope"
    assert decide(out("benign_standard_traffic"), "L4_FLOW", None, 0.99, TAX, env, {"pkts_per_s": 20.0})[0]["verdict"] == "ALLOW"
    assert decide(out("syn_flood_ddos"), "L4_FLOW", None, 0.99, TAX, env, flood)[0]["verdict"] == "BLOCK"


def test_flow_envelope_from_frozen_policy_yaml():
    # the frozen policy stores 2e-06, which PyYAML loads as a string: the check must still compare numbers
    pol = yaml.safe_load('flow_envelope: {"duration_s": [2e-06, 2994.234]}')
    assert isinstance(pol["flow_envelope"]["duration_s"][0], str)
    assert not outside_envelope({"duration_s": 1.5}, pol["flow_envelope"])
    assert outside_envelope({"duration_s": 5000.0}, pol["flow_envelope"])


def test_envelope_written_by_calibration_reloads_as_numbers():
    env = {"duration_s": [2e-06, 2994.234], "pkts_total": [2, 14807]}
    text = yaml.safe_dump(env, default_flow_style=True, width=10**9).strip()  # what eval.calibration writes
    back = yaml.safe_load("flow_envelope: " + text)["flow_envelope"]
    assert back == env and all(isinstance(x, (int, float)) for v in back.values() for x in v)


def test_scanner_user_agent_monitors_benign_and_keeps_block():
    ua = [["Host", "a"], ["User-Agent", "Mozilla/5.0 (compatible; Nuclei/v3.4.1)"]]
    d, diag = decide(out("benign_standard_traffic"), "L7_HTTP", None, 0.95, TAX, CAL, None, ua)
    validate_decision(d, TAX, SCHEMA)
    assert (d["verdict"], d["action"], d["rule_hint"]) == ("ABSTAIN", "log_and_monitor", "tier0_scanner_nuclei")
    assert diag["scanner_user_agent"] == "nuclei"
    d, _ = decide(out("sqli_classic"), "L7_HTTP", None, 0.95, TAX, CAL, None, [["User-Agent", "sqlmap/1.8"]])
    assert d["verdict"] == "BLOCK"  # the model's attack verdict stands
    d, _ = decide(out("benign_standard_traffic"), "L7_HTTP", None, 0.95, TAX, CAL, None, [["User-Agent", "Mozilla/5.0 Burpee garden app"]])
    assert d["verdict"] == "ALLOW"  # whole-word match only


def test_over_budget_request_is_monitored_not_guessed():
    from eval.policy import over_budget_decision
    d = over_budget_decision(POL)
    validate_decision(d, TAX, SCHEMA)
    assert (d["verdict"], d["action"], d["rule_hint"]) == ("ABSTAIN", "log_and_monitor", "over_token_budget")
    assert over_budget_decision({"reason_codes": {}})["rule_hint"] == "over_token_budget"  # v10's frozen policy has no key

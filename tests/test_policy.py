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


def test_zero_duration_is_below_exporter_resolution_not_outside():
    # NetFlow records milliseconds: a 2-packet flow of 0 ms lies inside a CIC-fitted [2e-06, ...] duration box
    env = {"duration_s": ["2e-06", 3113.0], "pkts_per_s": [0.03, 1e6], "bytes_per_s": [0.0, 3.1e7], "pkts_total": [2, 21107]}
    short = {"duration_s": 0, "pkts_total": 2, "bytes_total": 120}
    assert not outside_envelope(short, env)
    assert outside_envelope(short, env, resolution={})  # exact comparison: the behaviour before 19 Oct
    assert outside_envelope({**short, "pkts_total": 1}, env)  # the other bounds still apply
    assert outside_envelope({**short, "pkts_total": 5000}, env)  # 5,000 packets in < 1 ms: > 1e6 pkt/s, a burst
    assert outside_envelope({**short, "bytes_total": 40_000}, env)  # > 4e7 B/s
    assert outside_envelope({"duration_s": 1e-6, "pkts_total": 2}, env)  # a measured 1 us stays below 2 us
    assert outside_envelope(short, {"duration_s": [0.01, 3113.0]})  # [0, 1 ms) lies wholly below a 10 ms bound
    pol = {**CAL, "flow_envelope": env}
    assert decide(out("benign_standard_traffic"), "L4_FLOW", None, 0.99, TAX, pol, short)[0]["verdict"] == "ALLOW"
    d, diag = decide(out("benign_standard_traffic"), "L4_FLOW", None, 0.99, TAX, {**pol, "flow_envelope_resolution": {}}, short)
    assert d["verdict"] == "ABSTAIN" and diag["reason"] == "out_of_envelope"


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


def test_decision_view_attack_mass():
    from eval.policy import decision_view
    benign = [False, False, True]  # two attack classes, one benign
    P = [[0.50, 0.40, 0.10],  # split between attacks: top_class 0.5, attack_mass 0.9 on class 0
         [0.20, 0.25, 0.55],  # benign wins at 0.55 under both views
         [0.30, 0.10, 0.60]]
    conf, pred = decision_view(P, benign, "top_class")
    assert list(pred) == [0, 2, 2] and conf[0] == 0.5
    conf, pred = decision_view(P, benign, "attack_mass")
    assert list(pred) == [0, 2, 2] and abs(conf[0] - 0.9) < 1e-9 and abs(conf[1] - 0.55) < 1e-9 and abs(conf[2] - 0.6) < 1e-9


def test_frozen_policies_name_a_decision_view():
    from eval.policy import decision_view
    root = Path(__file__).resolve().parent.parent / "config"
    for f in ("policy.yaml", "policy_gemma.yaml", "policy_ensemble.yaml"):  # the scorer passes act_on to decision_view
        decision_view([[0.6, 0.4]], [False, True], yaml.safe_load(open(root / f)).get("act_on", "top_class"))


def test_ensemble_probs_is_the_tempered_mean_of_member_distributions():
    import numpy as np
    from eval.policy import ensemble_log_mean, ensemble_probs
    pol = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config/policy_ensemble.yaml"))
    assert [m["name"] for m in pol["ensemble"]["members"]] == ["qwen35_4b_v11", "gemma4_e2b_it_1ep"]
    a, b = np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.8, 0.1])
    p = ensemble_probs([a, b], {**pol, "temperature": 1.0})
    assert np.allclose(p, (a + b) / 2) and np.allclose(np.exp(ensemble_log_mean([a, b])), (a + b) / 2)
    hot = ensemble_probs([a, b], {**pol, "temperature": 0.5})  # T < 1 sharpens the mean
    assert hot.argmax() == p.argmax() and hot.max() > p.max() and np.isclose(hot.sum(), 1.0)
    with pytest.raises(AssertionError):
        ensemble_probs([a], pol)  # one distribution per member
    d, _ = decide(out("benign_standard_traffic"), "L7_HTTP", None, float(p.max()), TAX, pol)
    validate_decision(d, TAX, SCHEMA)


def test_ensemble_merge_matches_members_and_drops_unshared_records(tmp_path):
    import numpy as np
    from eval import calibration as CAL
    from eval.ensemble import merge
    classes = list(TAX["classes"])
    rng = np.random.default_rng(0)

    def member(name, ids, T):
        rows = [{"classes": classes, "mode": "fast"}]
        rows += [{"id": i, "split": "test", "kind": None, "label": classes[0], "scores": rng.normal(-10, 3, len(classes)).tolist()}
                 for i in ids]
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (tmp_path / f"{name}.yaml").write_text(f"temperature: {T}\n")
        return rows, (name, str(tmp_path / f"{name}.jsonl"), str(tmp_path / f"{name}.yaml"))
    ra, ma = member("a", ["r1", "r2", "r3"], 1.2)
    rb, mb = member("b", ["r2", "r3", "r4"], 0.8)
    head, counts = merge([ma, mb], str(tmp_path / "ens.jsonl"))
    assert counts["merged"] == {"test": 2} and counts["dropped"] == {"a": {"test": 1}, "b": {"test": 1}}
    got = {r["id"]: r for r in map(json.loads, open(tmp_path / "ens.jsonl").read().splitlines()[1:])}
    sa, sb = {r["id"]: r["scores"] for r in ra[1:]}, {r["id"]: r["scores"] for r in rb[1:]}
    for i in ("r2", "r3"):
        want = (CAL.softmax(sa[i], 1.2) + CAL.softmax(sb[i], 0.8)) / 2
        assert np.allclose(CAL.softmax(got[i]["scores"]), want, atol=1e-5)
    assert [m["temperature"] for m in head["ensemble"]["members"]] == [1.2, 0.8]

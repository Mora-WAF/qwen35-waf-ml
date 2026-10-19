"""Benchmark tooling of 19 Oct: Gemma window export, endpoint package metadata, the endpoint's return_probs path, and
the offline results pipeline (pack -> unpack -> re-decide -> Gemma -> ensemble -> compare) on real benchmark records
with synthetic scores. No model, no GPU."""
import base64
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import calibration as CAL  # noqa: E402
from eval import policy as POL  # noqa: E402
from training import prepare as P  # noqa: E402

TAX, _, _ = POL.load()
CLASSES = list(TAX["classes"])
BENIGN = np.array([TAX["classes"][c]["is_benign"] for c in CLASSES])


def _jwt():
    b64 = lambda s: base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")  # noqa: E731
    return b64('{"alg":"HS256"}') + "." + b64('{"sub":"a"}') + ".c2ln"


def _http(body="", jwt=False, n_headers=1):
    h = [["Host", "a.example"]] + [[f"X-H{i}", "v"] for i in range(n_headers - 1)]
    if jwt:
        h.append(["Authorization", "Bearer " + _jwt()])
    return {"schema_version": 1, "mode": "L7_HTTP", "http": {"method": "POST", "target": "/x", "headers": h,
                                                             "query": None, "cookies": None, "body": body}}


# ---- Gemma export: windows exactly as the endpoint's sensor contract ------------------------------------------------
def test_bench_rows_windows_views_and_over_budget():
    from training.gemma_tpu.data import bench_rows
    ids_of = lambda tel: list(range(len(json.dumps(tel)) // 4))  # noqa: E731  ~4 characters per token
    short = {"id": "gtw:a", "bench": "gotestwaf", "telemetry": _http("q=1", jwt=True)}
    long_ = {"id": "gtw:b", "bench": "gotestwaf", "telemetry": _http("x=" + "".join(f"{i:05d}" for i in range(4000)))}
    huge_headers = {"id": "gtw:c", "bench": "gotestwaf", "telemetry": _http("", n_headers=3000)}
    rows, man = bench_rows([short, long_, huge_headers], ids_of, budget=1500)
    assert man["gtw:a"] == {"windows": 1, "bench": "gotestwaf", "truncated": False}
    assert man["gtw:b"]["windows"] > 1 and man["gtw:b"]["truncated"]  # body windows, each scored
    assert man["gtw:c"]["windows"] == 0  # fits no budget: the endpoint answers over_budget
    ids = [r[0] for r in rows]
    assert ids[0] == "gtw:a#0" and all(i.startswith("gtw:b#") for i in ids[1:]) and len(ids) == 1 + man["gtw:b"]["windows"]
    assert all(len(t) <= 1500 for _, t in rows)
    # the decoded views are in the scored prompt (the build carried them: train/serve parity)
    want = ids_of(P.add_decoded_views(json.loads(json.dumps(short["telemetry"]))))
    assert rows[0][1] == want and len(want) > len(ids_of(short["telemetry"]))
    assert short["telemetry"]["http"].get("decoded_views") is None  # the record itself is not modified


def test_prompt_buckets_hold_prompt_plus_longest_answer(tmp_path):
    from training.gemma_tpu.data import BUCKETS, K, write_prompt_buckets
    rows = [("r#0", list(range(10))), ("s#0", list(range(BUCKETS[0] - K + 1))), ("t#0", list(range(BUCKETS[0] - K)))]
    counts = write_prompt_buckets(tmp_path, "bench", rows, {"r#0": 3, "t#0": -1}, pad_id=0)
    assert counts == {BUCKETS[0]: 2, BUCKETS[1]: 1}
    d = dict(np.load(tmp_path / f"bench_{BUCKETS[0]}.npz"))
    assert list(d["ids"]) == ["r#0", "t#0"] and list(d["label"]) == [3, -1] and list(d["prompt_len"]) == [10, BUCKETS[0] - K]
    assert d["tokens"].shape == (2, BUCKETS[0]) and (d["tgt_mask"] == 0).all() and set(d["split"]) == {"bench"}


# ---- endpoint package metadata ----------------------------------------------------------------------------------
def test_views_evidence_tells_a_build_with_views_from_one_without():
    from deploy.package_meta import views_evidence
    raw = [_http("q=1", jwt=True) for _ in range(50)] + [_http("q=2") for _ in range(50)]
    with_views = [P.add_decoded_views(json.loads(json.dumps(t))) for t in raw]
    assert views_evidence(with_views) == (True, 100, 0)  # v11-style build: re-applying adds nothing
    assert views_evidence(raw) == (False, 100, 50)  # v10-style build: every JWT record gains a view
    assert views_evidence([{"mode": "L4_FLOW", "flow": {}}]) == (False, 0, 0)  # no HTTP records: no evidence, off


def test_selfcheck_picks_modes_first_then_the_hardest():
    from deploy.package_meta import pick_selfcheck
    rows, scores = [], {}
    for i, mode in enumerate(["L7_HTTP"] * 6 + ["L4_FLOW"] * 3):
        tel = {"schema_version": 1, "mode": mode, "http": {}} if mode == "L7_HTTP" else {"schema_version": 1, "mode": mode, "flow": {}}
        rows.append({"id": f"r{i}", "messages": [{}, {"content": P.serialize_telemetry(tel, [])}]})
        scores[f"r{i}"] = [-1.0, -1.0 - (0.01 if i == 8 else 5.0)]  # r8: near tie
    picked = pick_selfcheck(rows, scores)
    assert {x["telemetry"]["mode"] for x in picked[:5]} == {"L7_HTTP", "L4_FLOW"} and "r8" in [x["id"] for x in picked]


# ---- the endpoint's predict_fn: return_probs and the shared decision path ---------------------------------------
@pytest.fixture
def inference(monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841  the handler imports torch at module level
    sys.path.insert(0, str(ROOT / "deploy/sagemaker"))
    monkeypatch.setenv("WAF_DECODED_VIEWS", "1")
    import importlib
    import inference as I
    I = importlib.reload(I)
    return I


def test_predict_fn_returns_the_deciding_distribution(inference, monkeypatch):
    I = inference
    pol = yaml.safe_load(open(ROOT / "config/policy.yaml"))
    rng = np.random.default_rng(1)
    S = {}

    def fake_scores(st, tel):  # one fixed score vector per request body
        return S.setdefault(tel["http"]["body"], (rng.normal(-12, 4, len(CLASSES))).tolist())
    monkeypatch.setattr(I, "_scores", fake_scores)
    monkeypatch.setattr(I, "_prompt_ids", lambda st, tel: list(range(len(json.dumps(tel)) // 4)))
    st = {"tax": TAX, "pol": pol, "classes": CLASSES, "selfcheck_max_diff": 0.0}
    recs = [_http("a=1"), _http("b=2", jwt=True)]
    res = I.predict_fn((recs, "sequential", True), st)
    assert res["decoded_views"] is True
    for tel, x in zip(recs, res["results"]):
        p = CAL.softmax(S[tel["http"]["body"]], float(pol["temperature"]))
        assert np.allclose(x["probs"], p, atol=1e-6) and x["temperature"] == float(pol["temperature"])
        d, diag, top = POL.decide_from_probs(p, tel, TAX, pol)
        assert x["decision"] == d and x["top"] == top
    plain = I.predict_fn((recs, "sequential"), st)["results"]  # two-element input (older callers): no probs
    assert all("probs" not in x for x in plain)
    assert I.input_fn(json.dumps({"records": recs, "return_probs": True}))[2] is True


def test_serving_json_sets_the_views_unless_overridden(tmp_path, monkeypatch, inference):
    I = inference
    monkeypatch.setattr(I, "HERE", tmp_path)
    monkeypatch.delenv("WAF_DECODED_VIEWS")
    assert I._decoded_views() is False  # no serving.json: the v10 packages
    (tmp_path / "serving.json").write_text(json.dumps({"decoded_views": True}))
    assert I._decoded_views() is True
    monkeypatch.setenv("WAF_DECODED_VIEWS", "0")
    assert I._decoded_views() is False


# ---- offline results pipeline ----------------------------------------------------------------------------------
@pytest.fixture
def bench(tmp_path, monkeypatch):
    from eval import public_bench as PB
    src = ROOT / "reports/benchmarks"
    gtw = [json.loads(l) for l in open(src / "gotestwaf.jsonl")][:40]
    unsw = [json.loads(l) for l in open(src / "unsw.jsonl")]
    unsw = [r for r in unsw if r["expected"] == "benign"][:20] + [r for r in unsw if r["expected"] != "benign"][:20]
    for name, rows in (("gotestwaf", gtw), ("unsw", unsw)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        shutil.copy(src / f"{name}.build.json", tmp_path / f"{name}.build.json")
    monkeypatch.setattr(PB, "OUT", tmp_path)
    return PB, gtw + unsw


def _scores_for(rid, seed):
    r = np.random.default_rng(abs(hash((rid, seed))) % 2**32)
    s = r.normal(-14, 4, len(CLASSES))
    s[r.integers(len(CLASSES))] = -0.05  # one confident class
    return s


def test_results_pipeline_redecide_gemma_ensemble_compare(bench, tmp_path):
    PB, recs = bench
    qpol = yaml.safe_load(open(ROOT / "reports/colab_a100v11/policy_v11.yaml"))
    gpol = yaml.safe_load(open(ROOT / "config/policy_gemma.yaml"))
    epol = yaml.safe_load(open(ROOT / "config/policy_ensemble.yaml"))
    Tq, Tg = float(qpol["temperature"]), float(gpol["temperature"])
    # pack -> (Colab) -> unpack
    PB.pack(["gotestwaf", "unsw"], tmp_path / "in.jsonl.gz")
    packed = [json.loads(l) for l in gzip.open(tmp_path / "in.jsonl.gz", "rt")]
    assert [r["id"] for r in packed] == [r["id"] for r in recs] and set(packed[0]) == {"id", "telemetry"}
    fp = []
    for r in recs:  # what colab/vllm_bench.py fp writes: the endpoint's result with return_probs
        p = CAL.softmax(_scores_for(r["id"], "q"), Tq)
        d, diag, top = POL.decide_from_probs(p, r["telemetry"], TAX, qpol)
        fp.append({"id": r["id"], "decision": d, "diagnostics": diag, "top": top, "probs": p.tolist(), "temperature": Tq})
    fp[0] = {"id": recs[0]["id"], "decision": POL.over_budget_decision(qpol), "diagnostics": {"reason": "over_budget"}}
    (tmp_path / "fp_out.jsonl").write_text("".join(json.dumps(x) + "\n" for x in fp))
    PB.unpack(tmp_path / "fp_out.jsonl", "v11")
    assert len(PB.load_results("gotestwaf", "v11")) == 40 and len(PB.load_results("unsw", "v11")) == 40
    with pytest.raises(AssertionError):
        PB.unpack(tmp_path / "fp_out.jsonl", "v11")  # raw results are never overwritten
    # re-decide under the same policy: identical decisions (exact path); under another temperature: re-tempered
    PB.redecide("unsw", "v11", "reports/colab_a100v11/policy_v11.yaml", "v11_same")
    a, b = PB.load_results("unsw", "v11"), PB.load_results("unsw", "v11_same")
    assert all(a[i]["decision"] == b[i]["decision"] for i in a)
    hot = {**qpol, "temperature": 0.5}
    (tmp_path / "hot.yaml").write_text(yaml.safe_dump(hot))
    PB.redecide("unsw", "v11", str(tmp_path / "hot.yaml"), "v11_hot")
    for x in PB.load_results("unsw", "v11_hot").values():
        assert np.allclose(x["probs"], CAL.softmax(_scores_for(x["id"], "q"), 0.5), atol=1e-5)
    with pytest.raises(AssertionError):  # no probs (v10-style) and no scoring temperature given
        (tmp_path / "unsw.v10x.results.jsonl").write_text("".join(json.dumps({k: v for k, v in x.items() if k not in ("probs", "temperature")}) + "\n"
                                                                  for x in a.values()))
        PB.redecide("unsw", "v10x", "reports/colab_a100v11/policy_v11.yaml", "v10y")
    # Gemma: window scores + manifest -> results; the window with the most attack mass decides
    man = {"budget": 2028, "records": {r["id"]: {"windows": 2 if r["id"] == recs[1]["id"] else 1, "bench": r["bench"],
                                                 "truncated": False} for r in recs}}
    man["records"][recs[2]["id"]]["windows"] = 0
    rows = [{"classes": CLASSES}]
    for r in recs:
        for k in range(man["records"][r["id"]]["windows"]):
            rows.append({"id": f"{r['id']}#{k}", "split": "bench", "kind": None, "label": None,
                         "scores": _scores_for(r["id"] + str(k), "g").tolist()})
    (tmp_path / "SCORES_bench.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (tmp_path / "man.json").write_text(json.dumps(man))
    for n in ("gotestwaf", "unsw"):
        PB.gemma_results(n, tmp_path / "SCORES_bench.jsonl", tmp_path / "man.json", "config/policy_gemma.yaml", "gemma")
    g = PB.load_results("gotestwaf", "gemma")
    assert g[recs[2]["id"]]["decision"]["rule_hint"] == "over_token_budget"
    w = [CAL.softmax(_scores_for(recs[1]["id"] + str(k), "g"), Tg) for k in range(2)]
    best = max(w, key=lambda q: 1 - q[BENIGN].sum())
    assert np.allclose(g[recs[1]["id"]]["probs"], best, atol=1e-6)
    # ensemble of the two members under the frozen ensemble policy
    for n in ("gotestwaf", "unsw"):
        PB.ensemble(n, "config/policy_ensemble.yaml", ["v11", "gemma"], "ensemble")
    e, q = PB.load_results("gotestwaf", "ensemble"), PB.load_results("gotestwaf", "v11")
    rid = recs[5]["id"]
    want = POL.ensemble_probs([np.array(q[rid]["probs"]), np.array(g[rid]["probs"])], epol)
    assert np.allclose(e[rid]["probs"], want, atol=1e-6)
    assert e[rid]["decision"] == POL.decide_from_probs(want, recs[5]["telemetry"], TAX, epol)[0]
    assert e[recs[0]["id"]]["decision"]["rule_hint"] == "over_token_budget"  # a member could not fit it
    assert e[recs[2]["id"]]["decision"]["rule_hint"] == "over_token_budget"
    full = PB.compare(["v11", "gemma", "ensemble"], str(tmp_path / "cmp.json"))
    assert full["gotestwaf"]["shared"] == 40 and full["unsw"]["shared"] == 40
    assert set(full["unsw"]["shared_all"]["ensemble"]) >= {"attack_n", "block_rate", "benign_n", "false_block_rate"}


def test_predict_fn_cached_serving_path_matches_the_evaluated_path(inference):
    """The default scoring ("cached": batch_class_logprobs) gives the per-record class_logprobs scores through the
    whole predict_fn (windows, views, decision). Tiny random Qwen3.5 on CPU; needs the pinned tokenizer."""
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from training import colab_helpers as H
    I = inference
    try:
        tok = P.load_tokenizer()
    except Exception as e:
        pytest.skip(f"tokenizer not available: {e}")
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
                            linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                            pad_token_id=tok.pad_token_id)
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    pol = yaml.safe_load(open(ROOT / "config/policy.yaml"))
    st = {"model": model, "tok": tok, "tax": TAX, "pol": pol, "classes": CLASSES, "selfcheck_max_diff": 0.0,
          "suf": H.candidate_suffixes(tok, CLASSES), "controls": P.control_strings(tok)}
    recs = [_http("a=1"), _http("q=" + "select " * 30, jwt=True), _http("x=2"),
            {"schema_version": 1, "mode": "L4_FLOW", "flow": {"features": {"duration_s": 0, "pkts_total": 2}}}]
    seq = I.predict_fn(([json.loads(json.dumps(r)) for r in recs], "sequential", True), st)["results"]
    cached = I.predict_fn(([json.loads(json.dumps(r)) for r in recs], "cached", True), st)["results"]
    assert I.input_fn(json.dumps({"records": recs}))[1] == "cached"  # the default
    for a, b in zip(seq, cached):
        assert np.abs(np.array(a["probs"]) - np.array(b["probs"])).max() < 1e-3
        assert a["decision"]["verdict"] == b["decision"]["verdict"]


def test_serving_json_sets_the_fit_contract(tmp_path, monkeypatch, inference):
    I = inference
    monkeypatch.setattr(I, "HERE", tmp_path)
    assert I._fit_contract() == 1  # v10 / v11 packages
    (tmp_path / "serving.json").write_text(json.dumps({"decoded_views": True, "fit_contract": 2}))
    assert I._fit_contract() == 2


def test_fit_contract_evidence_from_the_build():
    from deploy.package_meta import fit_contract
    assert fit_contract([_http("a=1"), {"mode": "L4_FLOW", "flow": {}}]) == 1
    t = _http("a=1")
    t["http"]["cookies_truncated"] = 3
    assert fit_contract([_http("a=1"), t]) == 2

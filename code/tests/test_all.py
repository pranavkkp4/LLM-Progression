"""Unit and committed-artifact integrity tests."""
import json
import hashlib
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp1_progression import fit_loglinear, days_since_2018  # noqa: E402
from exp2_cost import pareto_frontier, AA_EFFORT  # noqa: E402
from exp4_self_consistency import (  # noqa: E402
    build_prompt,
    canonicalize_answer,
    exact_mcnemar_p,
    extract_answer,
    is_correct,
    load_gsm8k,
    majority_vote,
    synthetic_problems,
    vote_sizes,
    wilson_interval,
)
from exp4_v1_improved import (  # noqa: E402
    build_messages,
    build_verifier_messages,
    candidate_rows,
    load_locked_split,
)
from exp5_confidence_model import (  # noqa: E402
    FEATURE_NAMES,
    record_features,
    run_analysis,
)


# ------------------------------------------------------- exp1: trend fitting
def test_fit_loglinear_recovers_growth():
    x = np.arange(0, 1000)
    y = np.exp(0.002 * x) * 5.0  # exact exponential
    slope, intercept, per_year, r2 = fit_loglinear(x, y)
    assert abs(slope - 0.002) < 1e-6
    assert r2 > 0.999
    assert abs(per_year - np.exp(0.002 * 365.25)) < 1e-6


def test_days_since_2018_monotonic():
    d = days_since_2018(pd.to_datetime(["2018-01-01", "2020-01-01"]))
    assert d[0] == 0 and d[1] > 700


# ------------------------------------------------------- exp2: Pareto front
def test_pareto_frontier_excludes_dominated():
    df = pd.DataFrame(AA_EFFORT, columns=["model", "effort", "index", "cost"])
    front = pareto_frontier(df)
    # every dominated point must have a frontier point that is >= smart, <= cost
    for _, r in df.iterrows():
        better = front[(front["index"] >= r["index"]) & (front.cost <= r.cost)]
        assert len(better) >= 1
    # documented property: Terra is fully dominated in the AA dataset
    assert "Terra" not in set(front.model)


def test_pareto_frontier_sorted_by_cost():
    df = pd.DataFrame(AA_EFFORT, columns=["model", "effort", "index", "cost"])
    front = pareto_frontier(df)
    assert list(front.cost) == sorted(front.cost)
    assert list(front["index"]) == sorted(front["index"])


# ------------------------------------------------- exp4: pure answer logic
def test_extract_answer_hash_format():
    assert extract_answer("Step 1... #### 42") == "42"
    assert extract_answer("math #### 1,234") == "1234"
    assert extract_answer("math #### 1,234.0") == "1234"
    assert extract_answer("no number here") is None
    assert extract_answer("The answer is 17.") == "17"
    assert extract_answer("3 + 4 = 7") == "7"


def test_majority_vote():
    assert majority_vote(["5", "5", "6"]) == "5"
    assert majority_vote(["5", "5.0", "6"]) == "5"
    assert majority_vote(["5", "6", "5", "6"]) in ("5", "6")  # tie: deterministic
    assert majority_vote([None, None]) is None
    assert majority_vote([]) is None


def test_is_correct():
    assert is_correct("42", "42")
    assert is_correct("42.0", "42")
    assert not is_correct("41", "42")
    assert not is_correct(None, "42")


def test_vote_sizes_and_small_sample_statistics():
    assert vote_sizes(3) == [1, 2, 3]
    assert vote_sizes(8) == [1, 2, 4, 8]
    assert canonicalize_answer("-0.50") == "-0.5"
    low, high = wilson_interval(1, 16)
    assert 0 < low < 6.25 < high < 40
    assert exact_mcnemar_p([False] * 5, [True] * 5) == 0.0625


def test_synthetic_problems_answers_check():
    for seed in range(10):
        probs = synthetic_problems(20, seed=seed)
        assert len(probs) == 20
        for p in probs:
            assert float(p["gold"]) == int(float(p["gold"]))  # integer answers
            assert int(p["gold"]) > 0


def test_build_prompt_contains_question():
    q = "What is 2+2?"
    assert q in build_prompt(q)
    assert "####" in build_prompt(q)


def test_improved_v1_prompts_and_candidates():
    question = "What is 2 + 2?"
    for style in ("chat-concise", "chat-fewshot", "chat-check"):
        messages = build_messages(question, style)
        assert messages[0]["role"] == "system"
        assert question in messages[1]["content"]
        assert "####" in messages[0]["content"] or "####" in messages[1]["content"]
    candidates = candidate_rows(
        ["4", "4.0", "5", None], ["first", "duplicate", "other", "missing"]
    )
    assert candidates == [
        {"answer": "4", "generation": "first"},
        {"answer": "4.0", "generation": "duplicate"},
        {"answer": "5", "generation": "other"},
    ]
    verifier = build_verifier_messages(question, candidates)
    assert "Every candidate may be wrong" in verifier[1]["content"]


def test_v1_splits_are_locked_and_disjoint():
    development, _ = load_locked_split("development")
    evaluation, _ = load_locked_split("evaluation")
    assert len(development) == 16
    assert len(evaluation) == 100
    development_ids = {problem["id"] for problem in development}
    evaluation_ids = {problem["id"] for problem in evaluation}
    assert development_ids.isdisjoint(evaluation_ids)
    original = json.loads(
        (ROOT / "results" / "self_consistency_checkpoint.json").read_text(
            encoding="utf-8")
    )
    assert [problem["id"] for problem in development] == original["problem_ids"]




def test_v1_frozen_evaluation_artifacts_are_complete():
    v1 = ROOT / "results" / "v1"
    result = json.loads(
        (v1 / "v1_final_results.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads(
        (v1 / "v1_final_checkpoint.json").read_text(encoding="utf-8")
    )
    records = [
        json.loads(line)
        for line in (v1 / "v1_final_traces.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    frozen_path = ROOT / "data" / "v1_frozen_config.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(
        (ROOT / "data" / "gsm8k_v1_splits.json").read_text(encoding="utf-8")
    )

    assert result["frozen_config_sha256"] == hashlib.sha256(
        frozen_path.read_bytes()
    ).hexdigest()
    assert result["split_file_sha256"] == frozen["split_file_sha256"]
    assert result["split"] == frozen["split"] == "evaluation"
    assert result["run_name"] == frozen["run_name"] == "v1_final"
    assert result["model_key"] == frozen["model_key"] == "qwen1.5b"
    assert result["prompt_style"] == frozen["prompt_style"] == "chat-fewshot"
    assert result["reported_selector"] == frozen["reported_selector"] == "majority"
    assert result["k"] == frozen["k"] == 4
    assert result["temperature"] == frozen["temperature"] == 0.4
    assert result["device"] == "cuda"
    assert result["n_problems"] == result["trace_records"] == len(records) == 100
    assert checkpoint["records"] == records
    assert result["problem_ids"] == split_manifest["evaluation_ids"]
    assert set(split_manifest["development_ids"]).isdisjoint(
        split_manifest["evaluation_ids"]
    )
    assert all(len(record["sample_generations"]) == 4 for record in records)
    assert all(len(record["extracted"]) == 4 for record in records)
    assert all(answer is not None for record in records for answer in record["extracted"])

    assert sum(record["greedy_correct"] for record in records) == 58
    assert sum(record["votes"]["4"]["correct"] for record in records) == 62
    assert sum(record["verifier"]["correct"] for record in records) == 62
    assert sum(record["oracle_any_sample_correct"] for record in records) == 79
    assert result["greedy_accuracy"] == 58.0
    assert result["selected_accuracy"] == result["majority_accuracy"]["4"] == 62.0
    assert result["verifier_accuracy"] == 62.0
    assert result["oracle_any_sample_accuracy"] == 79.0
    assert abs(result["mcnemar_exact_p_selected_vs_greedy"] - 0.480682373046875) < 1e-12


def test_confidence_features_do_not_use_reference_answers():
    trace_path = ROOT / "results" / "v1" / "v1_final_traces.jsonl"
    record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    altered = json.loads(json.dumps(record))
    altered["gold"] = "not-the-reference"
    altered["greedy_correct"] = not altered["greedy_correct"]
    altered["votes"]["4"]["correct"] = not altered["votes"]["4"]["correct"]
    altered["verifier"]["correct"] = not altered["verifier"]["correct"]
    altered["oracle_any_sample_correct"] = not altered["oracle_any_sample_correct"]
    assert record_features(record) == record_features(altered)


def test_ml_confidence_artifact_is_reproducible():
    v1 = ROOT / "results" / "v1"
    trace_path = v1 / "v1_final_traces.jsonl"
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    committed = json.loads(
        (v1 / "ml_confidence_results.json").read_text(encoding="utf-8")
    )
    recomputed = run_analysis(
        records, hashlib.sha256(trace_path.read_bytes()).hexdigest()
    )

    assert committed["analysis_role"] == (
        "post-evaluation exploratory confidence analysis"
    )
    assert committed["features"] == FEATURE_NAMES
    assert committed["input_trace_sha256"] == recomputed["input_trace_sha256"]
    assert committed["n_items"] == 100
    assert committed["positive_items"] == 62
    assert {item["fold"] for item in committed["items"]} == {1, 2, 3, 4, 5}
    assert [item["id"] for item in committed["items"]] == [
        record["id"] for record in records
    ]
    assert np.allclose(
        [item["oof_probability"] for item in committed["items"]],
        [item["oof_probability"] for item in recomputed["items"]],
        rtol=0,
        atol=1e-12,
    )
    for metric, value in committed["metrics"].items():
        assert abs(value - recomputed["metrics"][metric]) < 1e-12
    assert committed["metrics"]["roc_auc"] > 0.8
    assert committed["metrics"]["brier_score"] < (
        committed["constant_prevalence_baseline"]["brier_score"]
    )
    top_half = committed["selective_accuracy"][0]
    assert top_half["coverage"] == 0.5
    assert top_half["n_selected"] == 50
    assert top_half["correct"] == 47
    assert top_half["accuracy"] == 94.0


def test_paper_main_is_arxiv_self_contained():
    paper = ROOT.parent / "paper"
    source = (paper / "main.tex").read_text(encoding="utf-8")

    assert r"\input{" not in source
    assert "sections/" not in source
    assert "../code/" not in source
    assert r"\bibliography{references}" in source
    assert (paper / "main.bbl").is_file()
    assert (paper / "references.bib").is_file()

    figure_paths = re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source
    )
    assert set(figure_paths) == {
        "figures/architecture.pdf",
        "figures/results.png",
        "figures/calibration.png",
        "figures/robustness.pdf",
    }
    assert all((paper / path).is_file() for path in figure_paths)
    assert "Confidence-ranked set" in source


# --------------------------------------------------------------- data files
def test_data_files_parse():
    m = pd.read_csv(ROOT / "data" / "models_timeline.csv")
    assert {"model", "release_date", "benchmark", "score"} <= set(m.columns)
    assert len(m) >= 20
    f = pd.read_csv(ROOT / "data" / "frontier_2026.csv")
    assert "GPT-5.6 Luna" in set(f.model)
    rates = f.set_index("model")[["input_price", "output_price"]]
    assert tuple(rates.loc["GPT-5.6 Luna"]) == (1.0, 6.0)
    assert tuple(rates.loc["GPT-5.6 Terra"]) == (2.5, 15.0)
    p = pd.read_csv(ROOT / "data" / "price_history.csv")
    assert (p.input_price_per_1m > 0).all()
    gsm8k = pd.read_parquet(ROOT / "data" / "gsm8k_test.parquet")
    assert len(gsm8k) == 1319
    assert {"question", "answer"} <= set(gsm8k.columns)


def test_committed_live_results_have_complete_traces():
    results = json.loads(
        (ROOT / "results" / "self_consistency_results.json").read_text(
            encoding="utf-8")
    )
    raw_lines = (ROOT / "results" / "exp4_raw_generations.jsonl").read_text(
        encoding="utf-8").splitlines()
    records = [json.loads(line) for line in raw_lines]
    checkpoint = json.loads(
        (ROOT / "results" / "self_consistency_checkpoint.json").read_text(
            encoding="utf-8")
    )
    assert results["schema_version"] == checkpoint["schema_version"] == 2
    assert results["dataset"] == "gsm8k"
    assert results["n_problems"] == 16
    assert results["k"] == 8
    assert results["seed"] == 1234
    assert results["data_seed"] == 42
    assert results["max_new_tokens"] == 192
    assert results["trace_records"] == results["n_problems"] == len(raw_lines)
    assert checkpoint["records"] == records
    expected, source = load_gsm8k(results["n_problems"],
                                  seed=results["data_seed"])
    assert source == "gsm8k"
    assert [record["id"] for record in records] == [p["id"] for p in expected]
    assert [record["question"] for record in records] == [
        p["question"] for p in expected
    ]
    assert [record["gold"] for record in records] == [p["gold"] for p in expected]
    assert len({record["id"] for record in records}) == len(records)
    assert all(len(record["sample_gens"]) == results["k"]
               for record in records)
    assert all(str(results["k"]) in record["votes"] for record in records)

"""Unit tests for the analysis and experiment code."""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp1_progression import fit_loglinear, days_since_2018  # noqa: E402
from exp2_cost import pareto_frontier, AA_EFFORT  # noqa: E402
from exp4_self_consistency import (extract_answer, majority_vote, is_correct,  # noqa: E402
                                   synthetic_problems, build_prompt)


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
    assert extract_answer("no number here") is None
    assert extract_answer("The answer is 17.") == "17"
    assert extract_answer("3 + 4 = 7") == "7"


def test_majority_vote():
    assert majority_vote(["5", "5", "6"]) == "5"
    assert majority_vote(["5", "6", "5", "6"]) in ("5", "6")  # tie: deterministic
    assert majority_vote([None, None]) is None
    assert majority_vote([]) is None


def test_is_correct():
    assert is_correct("42", "42")
    assert is_correct("42.0", "42")
    assert not is_correct("41", "42")
    assert not is_correct(None, "42")


def test_synthetic_problems_answers_check():
    probs = synthetic_problems(20, seed=1)
    assert len(probs) == 20
    for p in probs:
        assert float(p["gold"]) == int(float(p["gold"]))  # integer answers
        assert int(p["gold"]) > 0


def test_build_prompt_contains_question():
    q = "What is 2+2?"
    assert q in build_prompt(q)
    assert "####" in build_prompt(q)


# --------------------------------------------------------------- data files
def test_data_files_parse():
    m = pd.read_csv(ROOT / "data" / "models_timeline.csv")
    assert {"model", "release_date", "benchmark", "score"} <= set(m.columns)
    assert len(m) >= 20
    f = pd.read_csv(ROOT / "data" / "frontier_2026.csv")
    assert "GPT-5.6 Luna" in set(f.model)
    p = pd.read_csv(ROOT / "data" / "price_history.csv")
    assert (p.input_price_per_1m > 0).all()

"""Exploratory confidence model for the frozen v1 GSM8K traces.

This post-evaluation analysis uses only signals available after inference and
before consulting the reference answer. Five-fold out-of-fold probabilities
keep every scored item separate from the fold used to fit its classifier.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import pathlib

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TRACES = ROOT / "code" / "results" / "v1" / "v1_final_traces.jsonl"
DEFAULT_OUTPUT = ROOT / "code" / "results" / "v1" / "ml_confidence_results.json"
FEATURE_NAMES = [
    "plurality_share",
    "normalized_vote_entropy",
    "distinct_answers",
    "greedy_agrees_plurality",
    "verifier_agrees_plurality",
    "mean_generation_chars",
    "std_generation_chars",
]
RANDOM_STATE = 20260811
N_SPLITS = 5


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record_features(record: dict) -> list[float]:
    """Return reference-free confidence features for one inference trace."""
    answers = record["extracted"]
    if len(answers) != 4 or any(answer is None for answer in answers):
        raise ValueError("confidence model requires four parsed sample answers")
    generations = record["sample_generations"]
    if len(generations) != 4:
        raise ValueError("confidence model requires four sample generations")
    counts = Counter(answers)
    proportions = np.asarray(list(counts.values()), dtype=float) / len(answers)
    entropy = float(-np.sum(proportions * np.log(proportions)) / math.log(len(answers)))
    plurality = record["votes"]["4"]["prediction"]
    lengths = np.asarray([len(text) for text in generations], dtype=float)
    return [
        max(counts.values()) / len(answers),
        entropy,
        float(len(counts)),
        float(record["greedy_prediction"] == plurality),
        float(record["verifier"]["prediction"] == plurality),
        float(lengths.mean()),
        float(lengths.std(ddof=0)),
    ]


def confidence_matrix(records: list[dict]) -> np.ndarray:
    return np.asarray([record_features(record) for record in records], dtype=float)


def new_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, l1_ratio=0.0, solver="liblinear", max_iter=1000),
    )


def fixed_coverage(y: np.ndarray, probabilities: np.ndarray, coverage: float) -> dict:
    count = int(round(len(y) * coverage))
    order = np.lexsort((np.arange(len(y)), -probabilities))
    selected = order[:count]
    return {
        "coverage": coverage,
        "n_selected": count,
        "correct": int(y[selected].sum()),
        "accuracy": float(100 * y[selected].mean()),
        "minimum_oof_probability": float(probabilities[selected].min()),
    }


def quantile_bins(y: np.ndarray, probabilities: np.ndarray, n_bins: int = 4) -> list[dict]:
    order = np.lexsort((np.arange(len(y)), probabilities))
    rows = []
    for index, selected in enumerate(np.array_split(order, n_bins), start=1):
        rows.append(
            {
                "bin": index,
                "n": int(len(selected)),
                "mean_oof_probability": float(probabilities[selected].mean()),
                "observed_accuracy": float(100 * y[selected].mean()),
            }
        )
    return rows


def run_analysis(
    records: list[dict],
    trace_sha256: str,
    input_trace: str = "code/results/v1/v1_final_traces.jsonl",
) -> dict:
    if len(records) != 100:
        raise ValueError("the exploratory analysis expects the 100-item v1 evaluation")
    x = confidence_matrix(records)
    y = np.asarray([record["votes"]["4"]["correct"] for record in records], dtype=int)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    probabilities = np.zeros(len(records), dtype=float)
    fold_ids = np.zeros(len(records), dtype=int)
    for fold, (train, test) in enumerate(splitter.split(x, y), start=1):
        model = new_model()
        model.fit(x[train], y[train])
        probabilities[test] = model.predict_proba(x[test])[:, 1]
        fold_ids[test] = fold

    full_model = new_model()
    full_model.fit(x, y)
    classifier = full_model.named_steps["logisticregression"]
    prevalence = float(y.mean())
    threshold_predictions = probabilities >= 0.5
    return {
        "schema_version": 1,
        "analysis_role": "post-evaluation exploratory confidence analysis",
        "input_trace": input_trace,
        "input_trace_sha256": trace_sha256,
        "target": "vote_at_4_exact_match_correct",
        "features": FEATURE_NAMES,
        "feature_policy": "reference-free signals available after inference",
        "model": {
            "estimator": "StandardScaler + L2 logistic regression",
            "C": 1.0,
            "solver": "liblinear",
            "scikit_learn_version": sklearn.__version__,
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "n_items": int(len(y)),
        "positive_items": int(y.sum()),
        "metrics": {
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "brier_score": float(brier_score_loss(y, probabilities)),
            "log_loss": float(log_loss(y, probabilities)),
            "accuracy_at_0_5": float(100 * accuracy_score(y, threshold_predictions)),
        },
        "constant_prevalence_baseline": {
            "probability": prevalence,
            "brier_score": float(brier_score_loss(y, np.full(len(y), prevalence))),
            "log_loss": float(log_loss(y, np.full(len(y), prevalence))),
            "majority_class_accuracy": float(100 * max(prevalence, 1 - prevalence)),
        },
        "selective_accuracy": [
            fixed_coverage(y, probabilities, 0.50),
            fixed_coverage(y, probabilities, 0.75),
            fixed_coverage(y, probabilities, 1.00),
        ],
        "probability_quartiles": quantile_bins(y, probabilities),
        "full_sample_standardized_coefficients": dict(
            zip(FEATURE_NAMES, classifier.coef_[0].tolist())
        ),
        "full_sample_intercept": float(classifier.intercept_[0]),
        "items": [
            {
                "id": record["id"],
                "fold": int(fold_ids[index]),
                "target": int(y[index]),
                "oof_probability": float(probabilities[index]),
            }
            for index, record in enumerate(records)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=pathlib.Path, default=DEFAULT_TRACES)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_bytes = args.traces.read_bytes()
    records = load_jsonl(args.traces)
    try:
        trace_label = args.traces.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        trace_label = str(args.traces.resolve())
    result = run_analysis(
        records, hashlib.sha256(trace_bytes).hexdigest(), trace_label
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_keys = (
        "analysis_role",
        "n_items",
        "positive_items",
        "metrics",
        "constant_prevalence_baseline",
        "selective_accuracy",
    )
    summary = {key: result[key] for key in summary_keys}
    print(json.dumps(summary, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

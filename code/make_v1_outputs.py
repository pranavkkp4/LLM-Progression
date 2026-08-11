"""Generate v1 paper tables and the locked-evaluation figure."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
V1 = RESULTS / "v1"
PAPER_FIGURES = ROOT / "paper" / "figures"
FROZEN = ROOT / "code" / "data" / "v1_frozen_config.json"
SPLITS = ROOT / "code" / "data" / "gsm8k_v1_splits.json"
FINAL_TRACES = V1 / "v1_final_traces.jsonl"


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return 100 * (center - margin), 100 * (center + margin)


def pct(value: float) -> str:
    return f"{value:.1f}\\%"


def interval(values: list[float] | tuple[float, float]) -> str:
    return f"[{values[0]:.1f}, {values[1]:.1f}]"


def validate(final: dict) -> None:
    frozen_sha = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    if final["frozen_config_sha256"] != frozen_sha:
        raise RuntimeError("final result does not match the frozen manifest")

    records = [
        json.loads(line)
        for line in FINAL_TRACES.read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != 100:
        raise RuntimeError("final JSONL must contain 100 records")
    recomputed = {
        "greedy": sum(record["greedy_correct"] for record in records),
        "majority": sum(
            record["votes"]["4"]["correct"] for record in records
        ),
        "verifier": sum(
            record["verifier"]["correct"] for record in records
        ),
        "oracle": sum(
            record["oracle_any_sample_correct"] for record in records
        ),
    }
    expected = {"greedy": 58, "majority": 62, "verifier": 62, "oracle": 79}
    if recomputed != expected:
        raise RuntimeError(f"final trace counts changed: {recomputed}")

    if final["n_problems"] != 100 or final["trace_records"] != 100:
        raise RuntimeError("final evaluation must contain 100 complete traces")
    if final["reported_selector"] != "majority" or final["k"] != 4:
        raise RuntimeError("unexpected primary selector")
    if final["selected_correct"] != final["majority_correct"]["4"]:
        raise RuntimeError("selected result does not equal vote-at-four primary")
    if final["valid_sample_answers"] != final["total_sample_answers"]:
        raise RuntimeError("not all sampled answers were parseable")
    splits = load(SPLITS)
    development = set(splits["development_ids"])
    evaluation = set(splits["evaluation_ids"])
    if len(development) != 16 or len(evaluation) != 100:
        raise RuntimeError("locked split sizes changed")
    if development & evaluation:
        raise RuntimeError("development and evaluation splits overlap")
    if final["problem_ids"] != splits["evaluation_ids"]:
        raise RuntimeError("final result is not in locked evaluation order")


def write_final_table(final: dict) -> None:
    rows = [
        (
            "Greedy",
            final["greedy_correct"],
            final["greedy_accuracy"],
            final["greedy_wilson_95"],
            "baseline",
        ),
        (
            "Sample @ 1",
            final["majority_correct"]["1"],
            final["majority_accuracy"]["1"],
            final["majority_wilson_95"]["1"],
            "diagnostic",
        ),
        (
            "Vote @ 2",
            final["majority_correct"]["2"],
            final["majority_accuracy"]["2"],
            final["majority_wilson_95"]["2"],
            "diagnostic",
        ),
        (
            r"Vote @ 4",
            final["majority_correct"]["4"],
            final["majority_accuracy"]["4"],
            final["majority_wilson_95"]["4"],
            "prespecified primary",
        ),
        (
            "Verifier",
            final["verifier_correct"],
            final["verifier_accuracy"],
            final["verifier_wilson_95"],
            "prespecified secondary",
        ),
        (
            "Oracle: any sample",
            final["oracle_any_sample_correct"],
            final["oracle_any_sample_accuracy"],
            wilson(final["oracle_any_sample_correct"], final["n_problems"]),
            "ceiling diagnostic",
        ),
    ]
    lines = [
        r"\begin{tabular}{lrrcl}",
        r"\toprule",
        r"Method & Correct & Accuracy & Wilson 95\% CI & Role \\",
        r"\midrule",
    ]
    for name, correct, accuracy, ci, role in rows:
        lines.append(
            f"{name} & {correct}/100 & {pct(accuracy)} & {interval(ci)} & {role} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (V1 / "v1_evaluation_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_development_table(legacy: dict, dev05: dict, dev15: dict,
                            final: dict) -> None:
    rows = [
        (
            "Legacy raw prompt, 0.5B (dev.)",
            legacy["n_problems"],
            legacy["greedy_correct"],
            legacy["self_consistency_correct"]["8"],
            "--",
            "--",
        ),
        (
            "Chat + few-shot, 0.5B (dev.)",
            dev05["n_problems"],
            dev05["greedy_correct"],
            dev05["majority_correct"]["4"],
            dev05["verifier_correct"],
            dev05["oracle_any_sample_correct"],
        ),
        (
            "Chat + few-shot, 1.5B (dev.)",
            dev15["n_problems"],
            dev15["greedy_correct"],
            dev15["majority_correct"]["4"],
            dev15["verifier_correct"],
            dev15["oracle_any_sample_correct"],
        ),
        (
            "Frozen 1.5B configuration (eval.)",
            final["n_problems"],
            final["greedy_correct"],
            final["majority_correct"]["4"],
            final["verifier_correct"],
            final["oracle_any_sample_correct"],
        ),
    ]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Configuration & $n$ & Greedy & Vote & Verifier & Oracle \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append("{} & {} & {} & {} & {} & {} \\\\".format(*row))
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (V1 / "v1_development_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_sentence(final: dict) -> None:
    text = (
        f"On the untouched 100-item evaluation split, greedy decoding solved "
        f"{final['greedy_correct']}/100 ({pct(final['greedy_accuracy'])}; "
        f"Wilson 95\\% CI {interval(final['greedy_wilson_95'])}), while the "
        f"prespecified four-sample plurality solved "
        f"{final['majority_correct']['4']}/100 "
        f"({pct(final['majority_accuracy']['4'])}; Wilson 95\\% CI "
        f"{interval(final['majority_wilson_95']['4'])}). The verifier also "
        f"solved {final['verifier_correct']}/100 "
        f"({pct(final['verifier_accuracy'])}). The primary paired exact "
        f"McNemar test gave $p={final['mcnemar_exact_p_selected_vs_greedy']:.3f}$."
    )
    (V1 / "v1_evaluation_sentence.tex").write_text(
        text + "\n", encoding="utf-8"
    )


def write_ml_table(ml: dict) -> None:
    if ml["analysis_role"] != "post-evaluation exploratory confidence analysis":
        raise RuntimeError("confidence results must remain explicitly exploratory")
    if ml["n_items"] != 100 or ml["positive_items"] != 62:
        raise RuntimeError("confidence result does not match the frozen evaluation")
    trace_sha256 = hashlib.sha256(FINAL_TRACES.read_bytes()).hexdigest()
    if ml["input_trace_sha256"] != trace_sha256:
        raise RuntimeError("confidence result does not match the frozen trace")
    coverage = [row["coverage"] for row in ml["selective_accuracy"]]
    if coverage != [0.5, 0.75, 1.0]:
        raise RuntimeError("unexpected selective-coverage levels")
    rows = [
        ("Top half", ml["selective_accuracy"][0]),
        ("Top three quarters", ml["selective_accuracy"][1]),
        ("All items", ml["selective_accuracy"][2]),
    ]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Confidence-ranked set & Coverage & Correct & Accuracy \\",
        r"\midrule",
    ]
    for name, row in rows:
        lines.append(
            f"{name} & {row['n_selected']}/100 & "
            f"{row['correct']}/{row['n_selected']} & "
            f"{pct(row['accuracy'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (V1 / "ml_confidence_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def make_figure(final: dict, ml: dict) -> None:
    labels = ["Greedy", "Sample\n@ 1", "Vote\n@ 2", "Vote\n@ 4", "Verifier", "Oracle\nany"]
    counts = [
        final["greedy_correct"],
        final["majority_correct"]["1"],
        final["majority_correct"]["2"],
        final["majority_correct"]["4"],
        final["verifier_correct"],
        final["oracle_any_sample_correct"],
    ]
    values = [float(value) for value in counts]
    cis = [
        final["greedy_wilson_95"],
        final["majority_wilson_95"]["1"],
        final["majority_wilson_95"]["2"],
        final["majority_wilson_95"]["4"],
        final["verifier_wilson_95"],
        wilson(final["oracle_any_sample_correct"], final["n_problems"]),
    ]
    errors = [
        [value - ci[0] for value, ci in zip(values, cis)],
        [ci[1] - value for value, ci in zip(values, cis)],
    ]
    colors = ["#64748b", "#93c5fd", "#60a5fa", "#2563eb", "#7c3aed", "#d97706"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    bars = ax.bar(
        labels, values, yerr=errors, capsize=4, color=colors,
        edgecolor="#1f2937", linewidth=0.7
    )
    ax.axhline(final["greedy_accuracy"], color="#475569", linestyle="--", linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_title("Locked GSM8K evaluation (n=100)")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 2.2,
            f"{value:.0f}%", ha="center", va="bottom", fontsize=9
        )
    confidence_ax = axes[1]
    selective = ml["selective_accuracy"]
    coverage_labels = ["Top 50%", "Top 75%", "All"]
    selective_values = [row["accuracy"] for row in selective]
    confidence_bars = confidence_ax.bar(
        coverage_labels,
        selective_values,
        color=["#059669", "#34d399", "#94a3b8"],
        edgecolor="#1f2937",
        linewidth=0.7,
    )
    confidence_ax.axhline(
        final["selected_accuracy"], color="#475569", linestyle="--", linewidth=1
    )
    confidence_ax.set_ylim(0, 100)
    confidence_ax.set_ylabel("Vote@4 accuracy (%)")
    confidence_ax.set_title("Out-of-fold confidence triage")
    confidence_ax.grid(axis="y", alpha=0.2)
    confidence_ax.text(
        0.98,
        0.04,
        f"AUC = {ml['metrics']['roc_auc']:.3f}\n"
        f"Brier = {ml['metrics']['brier_score']:.3f}",
        transform=confidence_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    for bar, value in zip(confidence_bars, selective_values):
        confidence_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "calibration.png", dpi=220)
    plt.close(fig)


def main() -> None:
    legacy = load(RESULTS / "self_consistency_results.json")
    dev05 = load(V1 / "dev_fewshot_t04_qwen05_results.json")
    dev15 = load(V1 / "dev_fewshot_t04_qwen15_results.json")
    ml = load(V1 / "ml_confidence_results.json")
    final = load(V1 / "v1_final_results.json")
    validate(final)
    write_final_table(final)
    write_development_table(legacy, dev05, dev15, final)
    write_ml_table(ml)
    write_sentence(final)
    make_figure(final, ml)
    print(V1 / "v1_evaluation_table.tex")
    print(V1 / "v1_development_table.tex")
    print(V1 / "v1_evaluation_sentence.tex")
    print(V1 / "ml_confidence_table.tex")
    print(PAPER_FIGURES / "calibration.png")


if __name__ == "__main__":
    main()

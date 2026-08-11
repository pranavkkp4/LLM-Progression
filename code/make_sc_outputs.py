"""Generate the paper figure + LaTeX table + sentence for experiment 4."""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT.parent / "paper" / "figures"
RES = ROOT / "results"


def main():
    with open(RES / "self_consistency_results.json", encoding="utf-8") as fh:
        result = json.load(fh)
    ks = sorted(int(k) for k in result["self_consistency_accuracy"])
    accuracies = [result["self_consistency_accuracy"][str(k)] for k in ks]
    intervals = [result["self_consistency_wilson_95"][str(k)] for k in ks]
    greedy = result["greedy_accuracy"]
    n = result["n_problems"]
    target_k = int(result["k"])
    target_accuracy = result["self_consistency_accuracy"][str(target_k)]
    greedy_interval = result["greedy_wilson_95"]
    target_interval = result["self_consistency_wilson_95"][str(target_k)]

    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axhspan(greedy_interval[0], greedy_interval[1], color="gray",
               alpha=0.12, lw=0)
    ax.axhline(greedy, ls="--", c="gray", lw=1.2,
               label=f"greedy decoding ({greedy:.1f}%)")
    lower_errors = [accuracy - interval[0]
                    for accuracy, interval in zip(accuracies, intervals)]
    upper_errors = [interval[1] - accuracy
                    for accuracy, interval in zip(accuracies, intervals)]
    ax.errorbar(ks, accuracies, yerr=[lower_errors, upper_errors], fmt="o-",
                c="#1f6fb4", capsize=3, label="sample-and-vote")
    for k, accuracy in zip(ks, accuracies):
        ax.annotate(f"{accuracy:.1f}", (k, accuracy),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    ax.set_xlabel("number of sampled chains $k$")
    ax.set_ylabel("accuracy (%)")
    upper_bounds = [interval[1] for interval in intervals]
    y_max = min(100, max(20, 1.1 * max([greedy_interval[1], *upper_bounds])))
    ax.set_ylim(0, y_max)
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_self_consistency.pdf")
    fig.savefig(FIG / "calibration.png", dpi=300)
    plt.close(fig)

    lines = [
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "decoding strategy & correct / $n$ & accuracy (\\%) "
        "& 95\\% Wilson interval \\\\",
        "\\midrule",
        f"greedy ($k=1$, no sampling) & {result['greedy_correct']} / {n} "
        f"& {greedy:.1f} & [{greedy_interval[0]:.1f}, "
        f"{greedy_interval[1]:.1f}] \\\\",
    ]
    for k, accuracy in zip(ks, accuracies):
        correct = result["self_consistency_correct"][str(k)]
        interval = result["self_consistency_wilson_95"][str(k)]
        lines.append(
            f"sample-and-vote, $k={k}$ & {correct} / {n} "
            f"& {accuracy:.1f} & [{interval[0]:.1f}, {interval[1]:.1f}] \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (RES / "self_consistency_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    delta = target_accuracy - greedy
    p_value = result["mcnemar_exact_p_at_k"]
    sentence = (
        f"In this {n}-item {result['dataset'].upper()} pilot with "
        f"{result['model'].split('/')[-1]}, greedy decoding achieved "
        f"{greedy:.1f}\\% ({result['greedy_correct']}/{n}; 95\\% Wilson "
        f"CI {greedy_interval[0]:.1f}--{greedy_interval[1]:.1f}) accuracy, "
        "while "
        f"sample-and-vote with $k={target_k}$ reached {target_accuracy:.1f}\\% "
        f"({result['self_consistency_correct'][str(target_k)]}/{n}; 95\\% "
        f"Wilson CI {target_interval[0]:.1f}--{target_interval[1]:.1f}) --- an "
        f"absolute gain of {delta:.1f} points (paired exact McNemar "
        f"$p={p_value:.4f}$)."
    )
    (RES / "self_consistency_sentence.tex").write_text(
        sentence + "\n", encoding="utf-8"
    )
    print(sentence)


if __name__ == "__main__":
    main()

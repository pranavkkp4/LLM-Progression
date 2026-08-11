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
    r = json.load(open(RES / "self_consistency_results.json"))
    ks = sorted((int(k) for k in r["self_consistency_accuracy"]))
    accs = [r["self_consistency_accuracy"][str(k)] for k in ks]
    greedy = r["greedy_accuracy"]

    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axhline(greedy, ls="--", c="gray", lw=1.2,
               label=f"greedy decoding ({greedy:.1f}%)")
    ax.plot(ks, accs, "o-", c="#1f6fb4", label="self-consistency (majority vote)")
    for k, a in zip(ks, accs):
        ax.annotate(f"{a:.1f}", (k, a), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("number of sampled chains $k$")
    ax.set_ylabel("accuracy (%)")
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_self_consistency.pdf")
    plt.close(fig)

    # LaTeX table
    lines = ["\\begin{tabular}{@{}lr@{}}", "\\toprule",
             "decoding strategy & accuracy (\\%) \\\\", "\\midrule",
             f"greedy ($k=1$, no sampling) & {greedy:.1f} \\\\"]
    for k, a in zip(ks, accs):
        lines.append(f"self-consistency, $k={k}$ & {a:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (RES / "self_consistency_table.tex").write_text("\n".join(lines) + "\n")

    # one-sentence summary used inline in the paper
    best_k = ks[accs.index(max(accs))]
    delta = max(accs) - greedy
    sentence = (
        f"On {r['n_problems']} {r['dataset']} problems with "
        f"{r['model'].split('/')[-1]}, greedy decoding achieved "
        f"{greedy:.1f}\\% accuracy, while self-consistency with $k={best_k}$ "
        f"sampled chains reached {max(accs):.1f}\\% --- an absolute gain of "
        f"{delta:+.1f} percentage points from inference-time compute alone."
    )
    (RES / "self_consistency_sentence.tex").write_text(sentence + "\n")
    print(sentence)


if __name__ == "__main__":
    main()

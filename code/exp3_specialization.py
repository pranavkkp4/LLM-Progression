"""Experiment 3: task specialization of 2026 frontier models.

Builds a model x task-category matrix from public benchmark results,
computes per-model specialization (spread of category-relative scores),
and evaluates a simple oracle router: how much better is
'pick the right model for each task' than any single model?
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT.parent / "paper" / "figures"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# Model x category matrix. Values are normalized 0-100 "share of best"
# computed below from the raw public numbers documented in data_sources.md.
RAW = {
    # category: {model: raw score}  (None = no public comparable number)
    "Agentic work\n(Agents' Last Exam)": {"GPT-5.6 Sol": 52.7, "Claude Opus 5": None,
                                          "Claude Fable 5": 40.5, "Kimi K3": None},
    "Repo coding\n(SWE-bench Pro)": {"GPT-5.6 Sol": 64.6, "Claude Opus 5": None,
                                     "Claude Fable 5": 80.0, "Kimi K3": None},
    "SWE tasks\n(SWE-bench Verified)": {"GPT-5.6 Sol": 96.2, "Claude Opus 5": 97.0,
                                         "Claude Fable 5": 95.0, "Kimi K3": 93.4},
    "Frontend\n(Arena Elo, 2026-08-10)": {"GPT-5.6 Sol": 1623,
                                           "Claude Opus 5": 1712,
                                           "Claude Fable 5": 1628,
                                           "Kimi K3": 1682},
    "Novel reasoning\n(ARC-AGI-3)": {"GPT-5.6 Sol": 7.78, "Claude Opus 5": 30.2,
                                     "Claude Fable 5": 20.0, "Kimi K3": None},
    "Knowledge work\n(GDPval-AA Elo)": {"GPT-5.6 Sol": 1747.8, "Claude Opus 5": None,
                                        "Claude Fable 5": 1759.6, "Kimi K3": 1686.0},
}

# OpenAI's GPT-5.6 launch table: every benchmark reported for BOTH
# GPT-5.6 Luna and GPT-5.5 (retrieved 2026-08 from openai.com/index/gpt-5-6).
LUNA_VS_55 = {
    # benchmark: (Luna, GPT-5.5, higher_is_better)
    "Agents' Last Exam": (50.3, 46.9, True),
    "GDPval-AA v2": (1591.8, 1493.7, True),
    "Consulting (internal)": (35.4, 31.3, True),
    "Big Finance Bench": (36.0, 49.0, True),
    "AA Intelligence Index": (51.2, 54.8, True),
    "AA Coding Agent Index": (74.6, 76.4, True),
    "SWE-bench Pro": (62.7, 59.4, True),
    "DeepSWE v1.1": (67.2, 67.0, True),
    "Terminal-Bench 2.1": (84.7, 85.6, True),
    "HealthBench Professional": (55.7, 49.5, True),
    "MMMU Pro (no tools)": (78.4, 81.2, True),
    "MMMU Pro (tools)": (79.5, 83.2, True),
    "gdp.pdf": (22.7, 26.0, True),
    "GPQA Diamond": (92.3, 93.6, True),
    "FrontierMath T1-3": (78.6, 85.3, True),
    "FrontierMath T4": (58.5, 72.5, True),
    "AutomationBench": (14.9, 12.9, True),
    "Toolathlon": (53.4, 55.6, True),
    "MRCR 256-512K": (41.3, 81.5, True),
    "MRCR 512K-1M": (41.3, 74.0, True),
    "GraphWalks 256k": (81.3, 73.7, True),
    "GraphWalks 1M": (51.2, 45.4, True),
    "ARC-AGI-3": (0.18, 0.43, True),
}


def main():
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    df = pd.DataFrame(RAW).T
    # normalize each category to % of the best model in that category
    norm = df.div(df.max(axis=1), axis=0) * 100
    norm.to_csv(RES / "specialization_matrix.csv")

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    im = ax.imshow(norm.values, cmap="viridis", vmin=80, vmax=100,
                   aspect="auto")
    ax.set_xticks(range(len(norm.columns)), norm.columns, fontsize=8)
    ax.set_yticks(range(len(norm.index)),
                  [i.replace("\n", " ") for i in norm.index], fontsize=8)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = norm.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if v < 93 else "black")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7,
                        color="lightgray")
    ax.set_title("Capability as % of category best (2026-07/08)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_specialization.pdf")
    fig.savefig(FIG / "robustness.pdf")
    plt.close(fig)

    # specialization spread: std of a model's normalized scores
    spread = norm.std(axis=0, skipna=True).sort_values(ascending=False)
    print("[specialization] within-model std of category-relative scores:")
    print(spread.round(1).to_string())

    # ---- oracle router: 14 benchmarks x 6 models from OpenAI's GPT-5.6
    # launch table (all cells public; see data_sources.md) ----
    B = {  # benchmark: [Sol, Terra, Luna, GPT-5.5, Fable 5, Opus 4.8]
        "Agents' Last Exam": [52.7, 50.4, 50.3, 46.9, 40.5, 45.2],
        "GDPval-AA v2": [1747.8, 1593.0, 1591.8, 1493.7, 1759.6, 1600.1],
        "Consulting (internal)": [43.2, 37.2, 35.4, 31.3, 35.5, 31.6],
        "AA Intelligence Index": [58.9, 55.0, 51.2, 54.8, 59.9, 55.7],
        "AA Coding Agent Index": [80.0, 77.4, 74.6, 76.4, 77.2, 72.5],
        "SWE-bench Pro": [64.6, 63.4, 62.7, 59.4, 80.0, 69.2],
        "DeepSWE v1.1": [72.7, 69.6, 67.2, 67.0, 69.7, 59.0],
        "Terminal-Bench 2.1": [88.8, 87.4, 84.7, 85.6, 83.1, 78.9],
        "GPQA Diamond": [94.6, 92.9, 92.3, 93.6, 92.6, 92.0],
        "FrontierMath T1-3": [89.0, 84.9, 78.6, 85.3, 87.0, 80.0],
        "FrontierMath T4": [83.0, 68.3, 58.5, 72.5, 87.8, 56.1],
        "AutomationBench": [18.1, 15.2, 14.9, 12.9, 17.4, 15.5],
        "Toolathlon": [58.0, 53.1, 53.4, 55.6, 61.7, 59.9],
        "gdp.pdf": [30.7, 24.7, 22.7, 26.0, 29.8, 22.5],
    }
    models = ["GPT-5.6 Sol", "GPT-5.6 Terra", "GPT-5.6 Luna", "GPT-5.5",
              "Claude Fable 5", "Claude Opus 4.8"]
    mat = pd.DataFrame(B, index=models).T
    nmat = mat.div(mat.max(axis=1), axis=0) * 100  # % of best per benchmark
    single_means = nmat.mean(axis=0).sort_values(ascending=False)
    oracle = nmat.max(axis=1).mean()
    print("\n[router] mean normalized score per single model (14 benchmarks):")
    print(single_means.round(1).to_string())
    print(f"[router] oracle router: {oracle:.1f} "
          f"(+{oracle - single_means.iloc[0]:.1f} vs best single)")
    nmat.to_csv(RES / "router_matrix.csv")
    with open(RES / "router_result.txt", "w") as fh:
        fh.write(f"best_single={single_means.iloc[0]:.2f}\n"
                 f"best_single_name={single_means.index[0]}\n"
                 f"oracle={oracle:.2f}\n")
        for m, v in single_means.items():
            fh.write(f"single:{m}={v:.2f}\n")

    # A *cheap* router: pick per benchmark the cheapest tier within one
    # normalized point of the best. Cost assumes five input tokens per output
    # token and uses official standard API rates as of 2026-08-10.
    price = {"GPT-5.6 Sol": 5 * 5 + 30,
             "GPT-5.6 Terra": 2.5 * 5 + 15,
             "GPT-5.6 Luna": 1 * 5 + 6,
             "GPT-5.5": 5 * 5 + 30,
             "Claude Fable 5": 10 * 5 + 50,
             "Claude Opus 4.8": 5 * 5 + 25}
    cheap_score, cheap_cost, oracle_cost = 0.0, 0.0, 0.0
    for b in nmat.index:
        row = nmat.loc[b]
        best = row.max()
        ok = [m for m in nmat.columns if row[m] >= best - 1.0]
        cheapest = min(ok, key=lambda m: price[m])
        cheap_score += row[cheapest]
        cheap_cost += price[cheapest]
        oracle_cost += price[row.idxmax()]
    cheap_score /= len(nmat)
    print(f"[cheap router] mean {cheap_score:.1f} at "
          f"{100 * cheap_cost / oracle_cost:.0f}% of oracle cost")
    with open(RES / "router_result.txt", "a") as fh:
        fh.write(f"cheap_router_score={cheap_score:.2f}\n"
                 f"cheap_router_cost_share={100 * cheap_cost / oracle_cost:.1f}\n")

    # two-vendor router: how much of the oracle does {Sol, Fable} capture?
    two = nmat[["GPT-5.6 Sol", "Claude Fable 5"]]
    two_oracle = two.max(axis=1).mean()
    print(f"[two-model router] Sol+Fable mean: {two_oracle:.1f} "
          f"(captures {100 * (two_oracle - single_means.iloc[0]) / (oracle - single_means.iloc[0]):.0f}% "
          f"of the full oracle gain)")
    with open(RES / "router_result.txt", "a") as fh:
        fh.write(f"two_model_router={two_oracle:.2f}\n")

    # ---- Luna (budget, 2026-07) vs GPT-5.5 (flagship, 2026-04) ----
    rows = []
    wins = ties = 0
    for bench, (l, g, hib) in LUNA_VS_55.items():
        winner = "Luna" if (l > g if hib else l < g) else (
            "tie" if l == g else "GPT-5.5")
        if winner == "Luna":
            wins += 1
        elif winner == "tie":
            ties += 1
        rows.append({"benchmark": bench, "luna": l, "gpt55": g,
                     "winner": winner})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "luna_vs_gpt55.csv", index=False)
    n = len(LUNA_VS_55)
    print(f"\n[Luna vs GPT-5.5] Luna wins {wins}, ties {ties}, loses "
          f"{n - wins - ties} of {n} shared benchmarks")
    prof = out[out.benchmark.isin([
        "Agents' Last Exam", "GDPval-AA v2", "Consulting (internal)",
        "SWE-bench Pro", "DeepSWE v1.1", "Terminal-Bench 2.1",
        "HealthBench Professional", "AutomationBench", "Toolathlon",
        "AA Coding Agent Index"])]
    pw = (prof.winner == "Luna").sum()
    print(f"[Luna vs GPT-5.5] on professional/agentic work: Luna wins "
          f"{pw}/{len(prof)}")


if __name__ == "__main__":
    main()

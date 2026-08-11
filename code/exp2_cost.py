"""Experiment 2: cost-capability analysis.

(a) Token-price decline 2020-2026.
(b) Cost-per-intelligence-point Pareto frontier for the GPT-5.6 family
    (Artificial Analysis all-effort dataset).
(c) 'Cheap-now vs flagship-then': cost to buy a given capability level.
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

# Artificial Analysis all-effort dataset (IgorWarzocha gist mirror of
# artificialanalysis.ai, retrieved 2026-07-11): model, effort, index, $/task
AA_EFFORT = [
    ("Luna", "low", 33, 0.04), ("Luna", "medium", 38, 0.05),
    ("Luna", "high", 46, 0.09), ("Luna", "xhigh", 49, 0.14),
    ("Luna", "max", 51, 0.21),
    ("Terra", "low", 40, 0.10), ("Terra", "medium", 46, 0.13),
    ("Terra", "high", 49, 0.24), ("Terra", "xhigh", 52, 0.33),
    ("Terra", "max", 55, 0.55),
    ("Sol", "low", 49, 0.20), ("Sol", "medium", 54, 0.31),
    ("Sol", "high", 56, 0.45), ("Sol", "xhigh", 58, 0.68),
    ("Sol", "max", 59, 1.04),
]


def pareto_frontier(df):
    """Points not dominated (higher intelligence at <= cost)."""
    pts = df.sort_values("cost").reset_index(drop=True)
    keep, best = [], -np.inf
    for _, r in pts.iterrows():
        if r["index"] > best:
            keep.append(True)
            best = r["index"]
        else:
            keep.append(False)
    return pts[keep]


def main():
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})

    # (a) token price decline
    p = pd.read_csv(ROOT / "data" / "price_history.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.scatter(p.date, p.input_price_per_1m, c="#1f6fb4", zorder=3)
    ax.set_yscale("log")
    lab = {"GPT-3 Davinci": (4, 4), "GPT-3.5 Turbo": (4, 4),
           "GPT-4": (4, 4), "GPT-4 Turbo": (4, 4), "GPT-4o": (-34, 8),
           "GPT-4o (reduced)": (-30, -16), "GPT-5": (-40, -14),
           "GPT-5.5": (-38, 8), "GPT-5.6 Sol": (7, -3),
           "GPT-5.6 Terra": (7, -5), "GPT-5.6 Luna": (-95, -4),
           "Claude Opus 5": (7, 7), "Claude Fable 5": (-25, 10),
           "Kimi K3": (-72, -14)}
    for _, r in p.iterrows():
        ax.annotate(r["model"], (r.date, r.input_price_per_1m),
                    textcoords="offset points",
                    xytext=lab.get(r["model"], (3, 4)), fontsize=6.5)
    ax.set_ylabel("input $ / 1M tokens (log)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_price_decline.pdf")
    plt.close(fig)

    gpt3 = p[p.model == "GPT-3 Davinci"].iloc[0].input_price_per_1m
    luna = p[p.model == "GPT-5.6 Luna"].iloc[0].input_price_per_1m
    print(f"[prices] input $/1M: GPT-3 (2020) ${gpt3:.2f} -> "
          f"GPT-5.6 Luna (2026) ${luna:.2f}: {gpt3 / luna:.0f}x decline")

    # (b) effort-cost frontier
    df = pd.DataFrame(AA_EFFORT, columns=["model", "effort", "index", "cost"])
    front = pareto_frontier(df)
    df.to_csv(RES / "aa_effort_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = {"Luna": "#2e8b57", "Terra": "#b8860b", "Sol": "#1f6fb4"}
    for m, g in df.groupby("model"):
        ax.scatter(g.cost, g["index"], c=colors[m], label=m, zorder=3)
    ax.plot(front.cost, front["index"], c="#c0392b", lw=1.2, ls="--",
            label="Pareto frontier")
    flab = {("Luna", "low"): (5, -4), ("Luna", "medium"): (5, -8),
            ("Luna", "high"): (5, -10), ("Luna", "xhigh"): (5, -12),
            ("Luna", "max"): (-58, -4), ("Sol", "medium"): (-62, 6),
            ("Sol", "high"): (-58, -14), ("Sol", "xhigh"): (-66, 4),
            ("Sol", "max"): (-45, 7)}
    for _, r in front.iterrows():
        ax.annotate(f"{r.model} {r.effort}", (r.cost, r["index"]),
                    textcoords="offset points",
                    xytext=flab.get((r.model, r.effort), (4, -6)), fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("cost per Intelligence Index task ($, log)")
    ax.set_ylabel("AA Intelligence Index")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_effort_frontier.pdf")
    plt.close(fig)
    print(f"[frontier] dominated settings: {len(df) - len(front)}/{len(df)}; "
          f"Terra never appears on the frontier: "
          f"{('Terra' not in set(front.model))}")

    # (c) cost to buy capability: Luna max vs Sol max
    lm = df[(df.model == "Luna") & (df.effort == "max")].iloc[0]
    sm = df[(df.model == "Sol") & (df.effort == "max")].iloc[0]
    share = lm["index"] / sm["index"] * 100
    cshare = lm.cost / sm.cost * 100
    print(f"[value] Luna max buys {share:.0f}% of Sol max intelligence "
          f"for {cshare:.0f}% of the task cost")


if __name__ == "__main__":
    main()

"""Experiment 1: Capability progression, 2018-2026.

Fits log-linear trends to three benchmark eras (GLUE/SuperGLUE, MMLU,
SWE-bench Verified) and estimates capability growth rates. All figures
are written to ../paper/figures/.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT.parent / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def days_since_2018(dates):
    delta = pd.to_datetime(dates) - pd.Timestamp("2018-01-01")
    return np.asarray(delta.dt.days if hasattr(delta, "dt") else delta.days)


def fit_loglinear(x, y):
    """Return slope (per year), implied gain per year (multiplicative), R^2."""
    slope, intercept = np.polyfit(x, np.log(y), 1)
    pred = slope * x + intercept
    ss_res = np.sum((np.log(y) - pred) ** 2)
    ss_tot = np.sum((np.log(y) - np.log(y).mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    per_year = np.exp(slope * 365.25)
    return slope, intercept, per_year, r2


def main():
    df = pd.read_csv(ROOT / "data" / "models_timeline.csv")
    df["release_date"] = pd.to_datetime(df["release_date"])

    # ---- SWE-bench Verified era (agentic coding) ----
    swe = df[df.benchmark.str.contains("SWE-bench")].copy()
    x = days_since_2018(swe.release_date)
    y = swe.score.values / (100 - swe.score.values)  # logit-ish: odds of solving
    slope, intercept, per_year, r2 = fit_loglinear(x, y)
    print(f"[SWE-bench Verified] solve-odds grow {per_year:.2f}x per year "
          f"(R^2={r2:.2f}, n={len(swe)})")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.scatter(swe.release_date, swe.score, c="#1f6fb4", zorder=3)
    xs = pd.date_range(swe.release_date.min(), swe.release_date.max(), 200)
    odds = np.exp(intercept + slope * days_since_2018(xs))
    ax.plot(xs, 100 * odds / (1 + odds), c="#c0392b", lw=1.5,
            label=f"log-linear fit, {per_year:.1f}x solve-odds/yr")
    lab_off = {"Claude 3.5 Sonnet (Oct)": (4, -10), "GPT-5": (4, 3),
               "Claude Opus 5": (-78, 5), "Kimi K3": (6, -16),
               "GPT-5.6 Sol": (6, -5), "GPT-5.5": (-52, -16),
               "Claude Fable 5": (-95, -6)}
    for _, r in swe.iterrows():
        if r["model"] in lab_off:
            ax.annotate(r["model"], (r.release_date, r.score),
                        textcoords="offset points",
                        xytext=lab_off[r["model"]], fontsize=7)
    ax.set_ylabel("SWE-bench Verified (%)")
    ax.set_ylim(40, 100)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_swebench_trend.pdf")
    plt.close(fig)

    # ---- MMLU era (knowledge) ----
    mmlu = df[df.benchmark.str.contains("MMLU")].copy()
    offsets = {"GPT-3": (4, 2), "GPT-3 ": (4, 2), "InstructGPT": (4, 2),
               "GPT-4": (-30, 4), "GPT-4o": (-40, 6),
               "Claude 3.5 Sonnet": (6, -16),
               "o1": (4, 4), "GPT-4.1": (4, -12)}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.scatter(mmlu.release_date, mmlu.score, c="#2e8b57", zorder=3)
    for _, r in mmlu.iterrows():
        ax.annotate(r["model"], (r.release_date, r.score),
                    textcoords="offset points",
                    xytext=offsets.get(r["model"], (4, 3)), fontsize=7)
    ax.axhline(89.8, ls="--", lw=1, c="gray")
    ax.text(mmlu.release_date.min(), 90.3, "estimated human-expert level (89.8%)",
            fontsize=7, color="gray")
    ax.set_ylabel("MMLU (%)")
    ax.set_ylim(38, 100)
    fig.tight_layout()
    fig.savefig(FIG / "fig_mmlu.pdf")
    plt.close(fig)

    # ---- Headline: years for budget tier to match prior flagship ----
    # GPT-5.6 Luna (2026-07, $0.20/$1.20) vs GPT-5.5 (2026-04, $5/$30)
    # and vs GPT-5 (2025-08 flagship).
    f = pd.read_csv(ROOT / "data" / "frontier_2026.csv")
    luna = f[f.model == "GPT-5.6 Luna"].iloc[0]
    g55 = f[f.model == "GPT-5.5"].iloc[0]
    benches = ["AA_Intelligence", "AA_CodingAgent", "SWEbench_Pro",
               "TerminalBench21", "ALE", "DeepSWE", "GDPval_AA"]
    wins = sum(luna[b] >= g55[b] for b in benches)
    print(f"[Luna vs GPT-5.5] Luna wins/ties on {wins}/{len(benches)} "
          f"shared benchmarks at 1/25th the output price")

    # derived overall SWE-bench Verified for Luna from Vals difficulty rows
    v = pd.read_csv(ROOT / "data" / "vals_swebench_difficulty.csv")
    counts = np.array([194, 261, 42, 3])
    v["overall"] = (v[["lt15min", "n1_15to60min", "n2_1to4hr", "gt4hr"]]
                    .values * counts).sum(1) / counts.sum()
    print(v[["model", "overall"]].round(1).to_string(index=False))
    v[["model", "overall"]].round(2).to_csv(ROOT / "results" / "swebench_overall_derived.csv",
                                            index=False)


if __name__ == "__main__":
    (ROOT / "results").mkdir(exist_ok=True)
    main()

# Data sources and retrieval dates

All web sources retrieved August 6–10, 2026.

## models_timeline.csv
- BERT GLUE 80.5: Devlin et al., NAACL 2019 (arXiv:1810.04805)
- RoBERTa GLUE 88.5: Liu et al. 2019 (arXiv:1907.11692)
- T5 SuperGLUE 90.3: Raffel et al., JMLR 2020
- GPT-3 SuperGLUE 71.8 few-shot: Brown et al., NeurIPS 2020
- GPT-3 MMLU 43.9 / GPT-4 MMLU 86.4: OpenAI GPT-4 Technical Report, 2023
- GPT-4o MMLU 88.7, o1 MMLU 91.8, GPT-4.1 MMLU 90.2: BenchLM MMLU mirror
  (benchlm.ai/benchmarks/mmlu), retrieved 2026-08-07
- SWE-bench Verified 2024–2025 rows: Anthropic/OpenAI announcements and
  swebench.com leaderboard (mini-SWE-agent harness), retrieved 2026-08
- SWE-bench Verified 2026 rows (Fable 5 95.0, Sol 96.2, K3 93.4, Opus 5 97.0):
  Vals AI independent harness (vals.ai/benchmarks/swebench), retrieved 2026-08-06
- GPT-5.5 85.1: vendor-reported (OpenAI release notes, via
  localaimaster.com/models/gpt-5-5), 2026-05

## frontier_2026.csv
- OpenAI models, shared comparison columns, and standard API prices:
  OpenAI GPT-5.6 launch tables (openai.com/index/gpt-5-6) and official API
  pricing/model documentation, re-verified 2026-08-10
- Vals AI SWE-bench Verified (independent): vals.ai/benchmarks/swebench
- Kimi K3 architecture, BrowseComp 91.2, and selected benchmark values:
  Moonshot AI's official Kimi K3 repository and technical report
- Frontend Arena Elo as of 2026-08-10 (Opus 5 1712, Kimi K3 1682,
  Fable 5 1628, Sol 1623): arena.ai live WebDev leaderboard; scores are
  preliminary and will move as votes accumulate
- Claude Opus 5 (SWE-bench 97.0, ARC-AGI-3 30.2, IMO 42/42, prices):
  Vals AI; ARC Prize (arcprize.org/results/anthropic-claude-opus-5);
  Anthropic launch post and system card, 2026-07-24
- GPT-5.6 standard API prices are $5/$30 (Sol), $2.50/$15 (Terra), and
  $1/$6 (Luna) per million input/output tokens; official OpenAI pricing,
  re-verified 2026-08-10

## price_history.csv
- 2020–2025 rows: deploybase.ai pricing history (2026-03-04)
- 2026 rows: OpenAI/Anthropic pricing pages via the reviews above

## vals_swebench_difficulty.csv
- Vals AI SWE-bench Verified per-difficulty rows (vals.ai/benchmarks/swebench),
  retrieved 2026-08-06. Task counts 194/261/42/3 as published there.

## AA effort dataset (hard-coded in exp2_cost.py)
- Artificial Analysis all-effort measurements for the GPT-5.6 family, mirrored
  at gist.github.com/IgorWarzocha (2026-07-11) and consistent with
  artificialanalysis.ai/articles/gpt-5-6-has-landed (2026-07-09)

## gsm8k_test.parquet
- OpenAI GSM8K test split (1,319 rows), mirrored from ModelScope dataset
  AI-ModelScope/gsm8k (main/test-00000-of-00001.parquet), retrieved 2026-08-10.
  License: MIT (OpenAI, 2021; arXiv:2110.14168)

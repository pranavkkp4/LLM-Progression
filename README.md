# From BERT to Frontier Agents — code, data, and paper

Companion archive to the preprint *"From BERT to Frontier Agents: Eight Years
of Language-Model Progress, the Collapse of the Capability–Cost Curve, and the
Rise of Task-Targeted Models"* (August 2026) and the op-ed in `op-ed/`.

## Layout

```
paper/            LaTeX source (arXiv-style, Tectonic) -> compiled PDF
  main.tex
  sections/       one file per section
  references.bib
  figures/        four submission figures generated from data/results
code/
  data/           curated datasets (CSVs + GSM8K parquet); see data_sources.md
  results/        machine-readable outputs of every run
  tests/          pytest unit tests
  exp1_progression.py       capability trend fits (SWE-bench, MMLU figures)
  exp2_cost.py              price decline + Pareto frontier of effort settings
  exp3_specialization.py    task-specialization heatmap, router oracle,
                            Luna-vs-GPT-5.5 head-to-head
  exp4_self_consistency.py  legacy 0.5B / 16-item pilot
  prepare_v1_splits.py      creates the locked dev/evaluation manifest
  exp4_v1_improved.py       chat prompts, sampling, verifier, full traces
  freeze_v1_config.py       freezes a development-selected configuration
  run_frozen_v1.py          runs/resumes the single 100-item evaluation
  make_v1_outputs.py        validates results and builds paper table/figure
  run_all.py                runs exp1-3 (tests and live runs are separate)
  summarize_v1_runs.py      tabulates development screens and comparisons
op-ed/oped.md     general-audience companion piece
```

## Reproducing the analysis (exp1–exp3, tests)

Requires Python 3.10+, `pandas`, `numpy`, `matplotlib`, `scipy`, `pytest`.

```bash
cd code
python run_all.py          # runs exp1-3 and regenerates figures/results
pytest tests/ -q           # unit and committed-artifact integrity tests
```

## Reproducing the frozen v1 experiment

The v1 workflow uses the original 16 GSM8K items only for development, locks
100 disjoint evaluation IDs before tuning, and evaluates the frozen
configuration once. The selected system is Qwen2.5-1.5B-Instruct with the
official chat template, four worked examples, four samples at temperature 0.4
and top-p 0.95, and plurality voting. A candidate verifier is reported as a
prespecified secondary selector.

The committed evaluation result is:

- greedy: **58/100 (58.0%)**, Wilson 95% CI 48.2--67.2%;
- vote at four, primary: **62/100 (62.0%)**, CI 52.2--70.9%;
- verifier, secondary: **62/100 (62.0%)**, CI 52.2--70.9%;
- any-sample oracle: **79/100 (79.0%)**, CI 70.0--85.8%;
- paired exact McNemar test, primary versus greedy: **p=0.481**.

The four-point observed gain is real for these saved predictions but is not
statistically significant on 100 items. The 79% oracle ceiling shows that
candidate selection, not candidate generation, is the largest measured
remaining opportunity.

1. Download model files for `Qwen/Qwen2.5-1.5B-Instruct` into
   `models/Qwen2.5-1.5B-Instruct/`. Model weights are intentionally ignored by
   Git (about 3.1 GB). The 0.5B files are also needed only to reproduce the
   development comparison.

   ```bash
   hf download Qwen/Qwen2.5-1.5B-Instruct \
     --local-dir models/Qwen2.5-1.5B-Instruct
   hf download Qwen/Qwen2.5-0.5B-Instruct \
     --local-dir models/Qwen2.5-0.5B-Instruct
   ```

2. Install runtime dependencies. CPU execution is supported; CUDA is selected
   automatically when the installed PyTorch build exposes a compatible GPU.
   The saved paper evaluation used PyTorch 2.10.0+cu128 on an RTX 3050 and
   records `device: cuda`.

   ```bash
   pip install transformers torch pyarrow pandas scipy matplotlib pytest
   ```

3. Reproduce or resume the frozen evaluation and regenerate paper outputs.

   ```bash
   python code/run_frozen_v1.py
   python code/make_v1_outputs.py
   pytest code/tests -q
   ```

   `run_frozen_v1.py` validates every scientific setting against
   `code/data/v1_frozen_config.json`. It requires all 100 locked items,
   checkpoints after each record, and resumes without discarding completed
   work. Use `--fresh` only when intentionally recomputing the full evaluation.

The committed machine-readable artifacts are under `code/results/v1/`:
development screens and confirmations, the frozen evaluation summary,
100-record checkpoint, and JSONL trace. `make_v1_outputs.py` verifies the
manifest hash, split sizes/order/disjointness, trace count, selector, and
answer parseability before writing the paper table and figure.


## Reproducing the legacy live experiment (exp4)

exp4 runs a real 0.5B-parameter language model on GSM8K. No GPU is required.

1. **Model weights (not committed, ~1 GB).** Download
   `Qwen/Qwen2.5-0.5B-Instruct` (config.json, generation_config.json,
   merges.txt, vocab.json, tokenizer.json, tokenizer_config.json,
   model.safetensors) into `models/Qwen2.5-0.5B-Instruct/`. Any Hugging Face
   or ModelScope mirror works. With the Hugging Face CLI:

   ```bash
   hf download Qwen/Qwen2.5-0.5B-Instruct model.safetensors \
     --local-dir models/Qwen2.5-0.5B-Instruct
   ```

   `code/dl_chunks.sh` alternatively automates the ModelScope
   download on servers that cap responses at 100 MiB:

   ```bash
   bash code/dl_chunks.sh     # assembles /tmp/model_full.safetensors
   ```

   The tokenizer/config files from this repo's
   `models/Qwen2.5-0.5B-Instruct/` directory are already committed; only
   `model.safetensors` needs downloading. If the full weights file is absent,
   the script falls back to `$QWEN_PATH` (default `/tmp/qwen`).

   **Checkpointing:** exp4 atomically writes complete per-problem records to
   `results/self_consistency_checkpoint.json` after every problem and rebuilds
   the JSONL trace from those records. Per-problem random seeds make a resumed
   run identical to an uninterrupted run. Rerun the same command to resume;
   `--fresh` restarts from zero.

2. **Data (committed).** `code/data/gsm8k_test.parquet` is the official
   GSM8K test split (1,319 problems; OpenAI, MIT license), mirrored from
   ModelScope dataset `AI-ModelScope/gsm8k`. The paper run validates the row
   count and columns and fails if this file cannot be read. Synthetic problems
   require the explicit `--allow-synthetic` flag and are for smoke tests only.

3. **Run.**

   ```bash
   pip install transformers torch pyarrow
   python exp4_self_consistency.py --n-problems 16 --k 8 \
     --max-new-tokens 192 --num-threads 8
   python make_sc_outputs.py   # regenerates figure + LaTeX table
   ```

   Runtime depends strongly on CPU count and generation speed. The committed
   run records seeds and all inference settings in the result JSON.

4. **Our committed outputs** are in `code/results/self_consistency_results.json`
   (plus generated `.tex` snippets consumed by the paper, the resumable
   checkpoint, and every raw generation in `exp4_raw_generations.jsonl`). In
   the clean committed run (8 CPU threads, bfloat16), greedy decoding solved
   **1/16 (6.25%)** and sample-and-vote at $k=8$ solved **2/16 (12.5%)**. The
   Wilson 95% intervals are 1.1--28.3% and 3.5--36.0%, respectively, and the
   paired exact McNemar test gives $p=1.0$. This is a reproducibility pilot,
   not a precise estimate of the intervention's population effect.

## Rebuilding the paper

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Notes and honest caveats

- Mid-2026 benchmark numbers mix vendor-reported and independent evaluations;
  provenance is labeled in the paper and in `code/data_sources.md`.
- `GPT-5.6 Luna`'s overall SWE-bench Verified score (92.9%) is our
  reconstruction from Vals AI's published per-difficulty rows
  (194/261/42/3 tasks), not a number Vals publishes directly.
- Model weights are fetched by script rather than committed (size); the
  committed checkpoint lets an interrupted exp4 run resume losslessly.

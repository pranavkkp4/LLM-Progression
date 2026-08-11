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
  exp4_self_consistency.py  LIVE experiment: greedy vs self-consistency on
                            Qwen2.5-0.5B-Instruct / GSM8K
  make_sc_outputs.py        figure + LaTeX table for exp4
  run_all.py                runs exp1-3 (tests and exp4 are separate)
op-ed/oped.md     general-audience companion piece
```

## Reproducing the analysis (exp1–exp3, tests)

Requires Python 3.10+, `pandas`, `numpy`, `matplotlib`, `scipy`, `pytest`.

```bash
cd code
python run_all.py          # runs exp1-3 and regenerates figures/results
pytest tests/ -q           # unit and committed-artifact integrity tests
```

## Reproducing the live experiment (exp4)

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

"""Experiment 4: test-time compute improves a small model — a live demo.

We take Qwen2.5-0.5B-Instruct (a model small enough to run on a laptop
CPU) and measure self-consistency (Wang et al., 2022) — sampling k
chain-of-thought solutions and taking a majority vote — against greedy
decoding in a small grade-school-math pilot.

This is the paper's live measurement of an inference-time intervention,
with paired outcomes and complete traces rather than an assumed effect.

Usage:
    python exp4_self_consistency.py --n-problems 16 --k 8

Data: the committed official GSM8K test parquet. Synthetic template problems
are available only when ``--allow-synthetic`` is explicitly requested for a
smoke test; paper results never fall back silently.
"""
import argparse
import json
import math
import pathlib
import random
import re
import time
from decimal import Decimal, InvalidOperation

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Local copies (download instructions in README.md):
MODEL_PATH = ROOT.parent / "models" / "Qwen2.5-0.5B-Instruct"
GSM8K_PARQUET = ROOT / "data" / "gsm8k_test.parquet"
SCHEMA_VERSION = 2


# ---------------------------------------------------------------- pure utils
def canonicalize_answer(value):
    """Normalize numerically equivalent answers before voting."""
    if value is None:
        return None
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def extract_answer(text):
    """Extract the final numeric answer from a chain-of-thought output."""
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return canonicalize_answer(m[-1])
    m = re.findall(r"(?:answer is|answer:|=)\s*\$?(-?[\d,]+(?:\.\d+)?)",
                   text.lower())
    if m:
        return canonicalize_answer(m[-1])
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return canonicalize_answer(nums[-1]) if nums else None


def majority_vote(answers):
    """Return the most frequent answer (ties broken by first occurrence)."""
    answers = [canonicalize_answer(a) for a in answers]
    answers = [a for a in answers if a is not None]
    if not answers:
        return None
    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    return max(counts, key=lambda a: (counts[a], -answers.index(a)))


FEWSHOT_PREFIX = (
    "Solve each problem with short steps, ending with '#### <answer>'.\n\n"
    "Problem: A robe takes 2 bolts of blue fiber and half that much white "
    "fiber. How many bolts in total does it take?\n"
    "Solution: Blue fiber: 2 bolts. White fiber: 2/2 = 1 bolt. "
    "Total: 2 + 1 = 3 bolts.\n#### 3\n\n"
    "Problem: Josh buys a house for $80,000. He puts in $50,000 of repairs, "
    "increasing the value by 150%. How much profit does he make?\n"
    "Solution: New value: 80,000 * 2.5 = 200,000. "
    "Cost: 80,000 + 50,000 = 130,000. Profit: 200,000 - 130,000 = 70,000.\n"
    "#### 70000\n\n"
    "Problem: Weng earns $12 an hour for babysitting. Yesterday, she just "
    "did 50 minutes of babysitting. How much did she earn?\n"
    "Solution: Per minute: 12/60 = $0.2. For 50 minutes: 0.2 * 50 = $10.\n"
    "#### 10\n\n"
    "Problem: James writes a 3-page letter to 2 different friends twice a "
    "week. How many pages does he write a year?\n"
    "Solution: Per week: 3 * 2 * 2 = 12 pages. Per year: 12 * 52 = 624 "
    "pages.\n#### 624\n\n"
)


def build_prompt(question):
    """Few-shot prompt that enforces short, parseable solutions."""
    return f"{FEWSHOT_PREFIX}Problem: {question}\nSolution:"


def truncate_solution(text):
    """Keep only the target problem's solution (model continues generating
    new problems after '#### <answer>')."""
    for stop in ("\nProblem:", "\n\nProblem:"):
        if stop in text:
            text = text.split(stop)[0]
    return text


def is_correct(pred, gold, tol=1e-4):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < tol
    except (TypeError, ValueError):
        return False


def vote_sizes(k):
    """Report standard prefixes without labeling more samples than exist."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return sorted({value for value in (1, 2, 4, k) if value <= k})


def wilson_interval(successes, total, z=1.96):
    """Return a Wilson 95% interval as percentages."""
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(
        p * (1 - p) / total + z**2 / (4 * total**2)
    ) / denominator
    return [100 * (center - half_width), 100 * (center + half_width)]


def exact_mcnemar_p(greedy, sampled):
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    if len(greedy) != len(sampled):
        raise ValueError("paired outcomes must have the same length")
    improved = sum((not g) and s for g, s in zip(greedy, sampled))
    regressed = sum(g and (not s) for g, s in zip(greedy, sampled))
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i)
               for i in range(min(improved, regressed) + 1))
    return min(1.0, 2 * tail / (2**discordant))


# ------------------------------------------------------------------ dataset
def load_gsm8k(n, seed=42, allow_synthetic=False):
    try:
        import pandas as pd
        if not GSM8K_PARQUET.exists():
            raise FileNotFoundError(GSM8K_PARQUET)
        df = pd.read_parquet(GSM8K_PARQUET)
        required = {"question", "answer"}
        if not required <= set(df.columns):
            raise ValueError(f"missing GSM8K columns: {required - set(df.columns)}")
        if len(df) != 1319:
            raise ValueError(f"expected 1,319 GSM8K test rows, found {len(df)}")
        df = df.sample(n=min(n, len(df)), random_state=seed)
        out = [{"id": f"gsm8k-test-{idx}", "question": r["question"],
                "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
               for idx, r in df.iterrows()]
        print(f"[data] loaded {len(out)} GSM8K test problems")
        return out, "gsm8k"
    except Exception as e:  # noqa: BLE001
        if not allow_synthetic:
            raise RuntimeError(
                "GSM8K could not be loaded. Install pyarrow and verify "
                f"{GSM8K_PARQUET}; pass --allow-synthetic only for a smoke test."
            ) from e
        print(f"[data] GSM8K load failed ({e}); using synthetic smoke-test set")
        return synthetic_problems(n, seed=seed), "synthetic"


def synthetic_problems(n, seed=0):
    """Deterministic template word problems with computed answers."""
    rng = random.Random(seed)
    probs = []
    for _ in range(n):
        kind = rng.randrange(4)
        if kind == 0:
            a, b = rng.randint(2, 12), rng.randint(3, 20)
            c = rng.randint(1, a - 1)
            q = (f"A farmer packs {a} crates with {b} apples in each crate. "
                 f"She sells {c} crates. How many apples does she have left?")
            gold = a * b - c * b
        elif kind == 1:
            a, b, c = rng.randint(5, 40), rng.randint(2, 15), rng.randint(2, 8)
            q = (f"Tom had {a} marbles. He bought {b} bags with {c} marbles "
                 f"in each bag. How many marbles does Tom have now?")
            gold = a + b * c
        elif kind == 2:
            a, b = rng.randint(3, 15), rng.randint(4, 12)
            c = rng.randint(2, min(a - 1, 8))
            q = (f"A baker makes {a} trays of {b} cookies. {c} trays burn "
                 f"and are thrown away. How many cookies remain?")
            gold = (a - c) * b
        else:
            a, b, c = rng.randint(10, 60), rng.randint(2, 10), rng.randint(2, 6)
            q = (f"A school prints {a} worksheets per class for {b} classes. "
                 f"Then it prints {c} extra worksheets for each of the {b} "
                 f"classes. How many worksheets were printed in total?")
            gold = a * b + c * b
        probs.append({"id": f"synthetic-{len(probs)}", "question": q,
                      "gold": str(gold)})
    return probs


CKPT = RES / "self_consistency_checkpoint.json"
RAW = RES / "exp4_raw_generations.jsonl"


def atomic_json_dump(value, path):
    """Write JSON without leaving a partial checkpoint on interruption."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2)
        fh.write("\n")
    temporary.replace(path)


def write_raw_records(records):
    """Rebuild the JSONL trace from the checkpoint's complete records."""
    temporary = RAW.with_suffix(RAW.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(RAW)


# -------------------------------------------------------------- experiment
def run(model, tok, problems, dataset, k, max_new_tokens=220, seed=1234):
    """Greedy vs self-consistency, with per-problem checkpointing so an
    interrupted run can resume with identical per-problem sampling."""
    import torch

    sizes = vote_sizes(k)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "k": k,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "problem_ids": [problem["id"] for problem in problems],
    }
    records = []
    if CKPT.exists():
        with open(CKPT, encoding="utf-8") as fh:
            state = json.load(fh)
        expected = {key: state.get(key) for key in metadata}
        if expected != metadata or "records" not in state:
            raise RuntimeError(
                "checkpoint does not match this run (or uses the legacy, "
                "trace-incomplete format); rerun with --fresh"
            )
        records = state["records"]
        for index, record in enumerate(records):
            if index >= len(problems) or record.get("id") != problems[index]["id"]:
                raise RuntimeError("checkpoint record order is invalid; use --fresh")
        write_raw_records(records)
        print(f"[resume] {len(records)} complete problem records already done")
    t0 = time.time()
    for i, p in enumerate(problems):
        if i < len(records):
            continue
        inputs = tok(build_prompt(p["question"]), return_tensors="pt")

        # greedy baseline
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        gen = truncate_solution(
            tok.decode(out[0][inputs.input_ids.shape[1]:],
                       skip_special_tokens=True))
        greedy_prediction = extract_answer(gen)
        greedy_correct = is_correct(greedy_prediction, p["gold"])
        greedy_gen = gen

        # k sampled chains
        torch.manual_seed(seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + i)
        answers = []
        raw_gens = []
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=0.8, top_p=0.95,
                                 num_return_sequences=k,
                                 pad_token_id=tok.eos_token_id)
        for j in range(k):
            gen = truncate_solution(
                tok.decode(out[j][inputs.input_ids.shape[1]:],
                           skip_special_tokens=True))
            raw_gens.append(gen)
            answers.append(extract_answer(gen))
        record = {
            "i": i,
            "id": p["id"],
            "question": p["question"],
            "gold": p["gold"],
            "greedy_gen": greedy_gen,
            "greedy_prediction": greedy_prediction,
            "greedy_correct": greedy_correct,
            "sample_gens": raw_gens,
            "extracted": answers,
            "votes": {
                str(kk): {
                    "prediction": majority_vote(answers[:kk]),
                    "correct": is_correct(majority_vote(answers[:kk]), p["gold"]),
                }
                for kk in sizes
            },
        }
        records.append(record)
        state = {**metadata, "records": records}
        atomic_json_dump(state, CKPT)
        write_raw_records(records)
        el = time.time() - t0
        greedy_so_far = [r["greedy_correct"] for r in records]
        sampled_so_far = [r["votes"][str(k)]["correct"] for r in records]
        print(f"  [{i + 1}/{len(problems)}] greedy="
              f"{100 * sum(greedy_so_far) / len(records):.1f}% "
              f"sc@{k}={100 * sum(sampled_so_far) / len(records):.1f}% "
              f"({el:.0f}s)", flush=True)
    return records, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--num-threads", type=int, default=2)
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="permit synthetic data for smoke tests only")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any checkpoint and start over")
    args = ap.parse_args()

    if args.n_problems < 1 or args.k < 1 or args.max_new_tokens < 1:
        ap.error("n-problems, k, and max-new-tokens must be positive")
    if args.num_threads < 1:
        ap.error("num-threads must be positive")
    if args.fresh:
        for path in (CKPT, RAW):
            if path.exists():
                path.unlink()

    problems, source = load_gsm8k(args.n_problems, seed=args.data_seed,
                                  allow_synthetic=args.allow_synthetic)

    import os
    import torch
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    index = MODEL_PATH / "model.safetensors.index.json"
    weights = MODEL_PATH / "model.safetensors"
    if index.exists():
        model_path = str(MODEL_PATH)          # sharded copy (works everywhere)
    elif weights.exists() and weights.stat().st_size == 988097824:
        model_path = str(MODEL_PATH)          # full single-file copy
    else:
        model_path = os.environ.get("QWEN_PATH", "/tmp/qwen")
    print(f"[model] loading from {model_path}")
    print(f"[run] seed={args.seed}, data_seed={args.data_seed}, "
          f"threads={args.num_threads}, k={args.k}")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 dtype="bfloat16")
    model.eval()

    records, sizes = run(model, tok, problems, source, args.k,
                         args.max_new_tokens, args.seed)

    n = len(problems)
    greedy = [record["greedy_correct"] for record in records]
    votes = {
        kk: [record["votes"][str(kk)]["correct"] for record in records]
        for kk in sizes
    }
    greedy_successes = sum(greedy)
    sampled_successes = {kk: sum(values) for kk, values in votes.items()}
    result = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_ID,
        "dataset": source,
        "n_problems": n,
        "k": args.k,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.8,
        "top_p": 0.95,
        "num_threads": args.num_threads,
        "greedy_correct": greedy_successes,
        "greedy_accuracy": 100 * greedy_successes / n,
        "greedy_wilson_95": wilson_interval(greedy_successes, n),
        "self_consistency_correct": {
            str(kk): sampled_successes[kk] for kk in sizes
        },
        "self_consistency_accuracy": {
            str(kk): 100 * sampled_successes[kk] / n for kk in sizes
        },
        "self_consistency_wilson_95": {
            str(kk): wilson_interval(sampled_successes[kk], n) for kk in sizes
        },
        "mcnemar_exact_p_at_k": exact_mcnemar_p(greedy, votes[args.k]),
        "trace_records": len(records),
    }
    atomic_json_dump(result, RES / "self_consistency_results.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Experiment 4: test-time compute improves a small model — a live demo.

We take Qwen2.5-0.5B-Instruct (a model small enough to run on a laptop
CPU) and show that self-consistency (Wang et al., 2022) — sampling k
chain-of-thought solutions and taking a majority vote — materially
raises accuracy on grade-school math word problems over greedy decoding.

This is the paper's "suggestions with proof" experiment: the capability
of a fixed model is not fixed at inference time; how you spend compute
at test time changes what the model can do.

Usage:
    python exp4_self_consistency.py --n-problems 16 --k 8

Data: GSM8K test set (downloaded via ModelScope). If the download is
unavailable, the script falls back to a deterministic synthetic set of
template word problems with programmatically computed answers.
"""
import argparse
import json
import pathlib
import random
import re
import time

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Local copies (download instructions in README.md):
MODEL_PATH = ROOT.parent / "models" / "Qwen2.5-0.5B-Instruct"
GSM8K_PARQUET = ROOT / "data" / "gsm8k_test.parquet"


# ---------------------------------------------------------------- pure utils
def extract_answer(text):
    """Extract the final numeric answer from a chain-of-thought output."""
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "").strip()
    m = re.findall(r"(?:answer is|answer:|=)\s*\$?(-?[\d,]+(?:\.\d+)?)",
                   text.lower())
    if m:
        return m[-1].replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def majority_vote(answers):
    """Return the most frequent answer (ties broken by first occurrence)."""
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


# ------------------------------------------------------------------ dataset
def load_gsm8k(n, seed=42):
    try:
        import pandas as pd
        df = pd.read_parquet(GSM8K_PARQUET)
        df = df.sample(n=min(n, len(df)), random_state=seed)
        out = [{"question": r["question"],
                "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
               for _, r in df.iterrows()]
        print(f"[data] loaded {len(out)} GSM8K test problems")
        return out, "gsm8k"
    except Exception as e:  # noqa: BLE001
        print(f"[data] GSM8K load failed ({e}); using synthetic set")
        return synthetic_problems(n), "synthetic"


def synthetic_problems(n, seed=0):
    """Deterministic template word problems with computed answers."""
    rng = random.Random(seed)
    probs = []
    for _ in range(n):
        kind = rng.randrange(4)
        if kind == 0:
            a, b, c = rng.randint(2, 12), rng.randint(3, 20), rng.randint(2, 9)
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
        probs.append({"question": q, "gold": str(gold)})
    return probs


CKPT = RES / "self_consistency_checkpoint.json"


# -------------------------------------------------------------- experiment
def run(model, tok, problems, k, max_new_tokens=220):
    """Greedy vs self-consistency, with per-problem checkpointing so an
    interrupted run can resume (see --resume)."""
    import torch

    greedy_correct, votes, done = [], {kk: [] for kk in [1, 2, 4, k]}, 0
    if CKPT.exists():
        st = json.load(open(CKPT))
        if st["k"] == k and st["n_problems"] == len(problems):
            greedy_correct = st["greedy"]
            votes = {int(kk): v for kk, v in st["votes"].items()}
            done = len(greedy_correct)
            print(f"[resume] {done} problems already done")
    t0 = time.time()
    for i, p in enumerate(problems):
        if i < done:
            continue
        inputs = tok(build_prompt(p["question"]), return_tensors="pt")

        # greedy baseline
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        gen = truncate_solution(
            tok.decode(out[0][inputs.input_ids.shape[1]:],
                       skip_special_tokens=True))
        greedy_correct.append(is_correct(extract_answer(gen), p["gold"]))
        greedy_gen = gen

        # k sampled chains
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
        # append raw generations for full transparency / inspection
        with open(RES / "exp4_raw_generations.jsonl", "a") as fh:
            fh.write(json.dumps({"i": i, "question": p["question"],
                                 "gold": p["gold"],
                                 "greedy_gen": greedy_gen,
                                 "sample_gens": raw_gens,
                                 "extracted": answers}) + "\n")
        for kk in votes:
            votes[kk].append(is_correct(majority_vote(answers[:kk]),
                                        p["gold"]))
        json.dump({"k": k, "n_problems": len(problems),
                   "greedy": greedy_correct,
                   "votes": {str(kk): v for kk, v in votes.items()}},
                  open(CKPT, "w"))
        el = time.time() - t0
        print(f"  [{i + 1}/{len(problems)}] greedy="
              f"{100 * sum(greedy_correct) / len(greedy_correct):.1f}% "
              f"sc@{k}={100 * sum(votes[k]) / len(votes[k]):.1f}% "
              f"({el:.0f}s)", flush=True)
    return greedy_correct, votes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any checkpoint and start over")
    args = ap.parse_args()

    if args.fresh and CKPT.exists():
        CKPT.unlink()

    problems, source = load_gsm8k(args.n_problems)

    import os
    import torch
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 2)
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
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 dtype="bfloat16")
    model.eval()

    greedy, votes = run(model, tok, problems, args.k, args.max_new_tokens)

    n = len(problems)
    result = {
        "model": MODEL_ID, "dataset": source, "n_problems": n, "k": args.k,
        "greedy_accuracy": 100 * sum(greedy) / n,
        "self_consistency_accuracy": {str(kk): 100 * sum(v) / len(v)
                                      for kk, v in votes.items()},
    }
    with open(RES / "self_consistency_results.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

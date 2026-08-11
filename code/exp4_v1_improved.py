"""Improved, leakage-controlled v1 GSM8K inference experiment.

The original 16-item raw-prompt run remains unchanged and becomes development
data. This script uses locked splits, Qwen's chat template, configurable
prompts and sampling, and a candidate verifier. Every run has isolated atomic
checkpoints and complete JSONL traces.
"""
import argparse
import json
import pathlib
import re
import time

from exp4_self_consistency import (
    FEWSHOT_PREFIX,
    GSM8K_PARQUET,
    atomic_json_dump,
    exact_mcnemar_p,
    extract_answer,
    is_correct,
    majority_vote,
    truncate_solution,
    vote_sizes,
    wilson_interval,
)

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results" / "v1"
SPLIT_PATH = ROOT / "data" / "gsm8k_v1_splits.json"
SCHEMA_VERSION = 1

MODEL_SPECS = {
    "qwen0.5b": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "path": ROOT.parent / "models" / "Qwen2.5-0.5B-Instruct",
    },
    "qwen1.5b": {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "path": ROOT.parent / "models" / "Qwen2.5-1.5B-Instruct",
    },
}
PROMPT_STYLES = ("chat-concise", "chat-fewshot", "chat-check")


def load_locked_split(name, limit=None):
    """Load one pre-registered split in its committed ID order."""
    import pandas as pd

    splits = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if splits.get("parquet_sha256") != sha256_text(GSM8K_PARQUET):
        raise RuntimeError("GSM8K parquet hash does not match split manifest")

    key = f"{name}_ids"
    if key not in splits:
        raise ValueError(f"unknown split: {name}")
    identifiers = splits[key]
    if limit is not None:
        if limit < 1 or limit > len(identifiers):
            raise ValueError(f"limit must be within 1..{len(identifiers)}")
        identifiers = identifiers[:limit]

    frame = pd.read_parquet(GSM8K_PARQUET)
    if len(frame) != 1319:
        raise ValueError(f"expected 1,319 GSM8K rows, found {len(frame)}")
    problems = []
    for identifier in identifiers:
        index = int(identifier.removeprefix("gsm8k-test-"))
        row = frame.loc[index]
        problems.append({
            "id": identifier,
            "question": row["question"],
            "gold": row["answer"].split("####")[-1].strip().replace(",", ""),
        })
    return problems, splits


def build_messages(question, style):
    """Return system/user messages for a named, auditable prompt style."""
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style: {style}")
    common = (
        "You are a careful grade-school mathematics solver. Use exact "
        "arithmetic, keep the reasoning concise, and end with exactly "
        "'#### <number>'."
    )
    if style == "chat-concise":
        system = common
        user = f"Solve this problem:\n\n{question}"
    elif style == "chat-fewshot":
        system = common
        user = f"{FEWSHOT_PREFIX}Problem: {question}\nSolution:"
    else:
        system = (
            f"{common} Before the final answer: (1) translate the quantities "
            "into equations, (2) compute them, and (3) independently check "
            "that the result answers the question and has the right units."
        )
        user = f"Solve and verify this problem:\n\n{question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def encode_chat(tokenizer, messages):
    """Apply the model's official instruction template."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt, return_tensors="pt"), prompt


def decode_completion(tokenizer, output, prompt_length):
    text = tokenizer.decode(
        output[prompt_length:], skip_special_tokens=True
    )
    return truncate_solution(text).strip()


def candidate_rows(answers, generations):
    """Deduplicate candidate answers while retaining one supporting trace."""
    rows = []
    seen = set()
    for answer, generation in zip(answers, generations):
        if answer is None or answer in seen:
            continue
        seen.add(answer)
        rows.append({"answer": answer, "generation": generation})
    return rows


def build_verifier_messages(question, candidates):
    """Ask the model to solve independently and select a supported candidate."""
    rendered = []
    for index, candidate in enumerate(candidates, start=1):
        solution = candidate["generation"][:800]
        rendered.append(
            f"Candidate {index}\nProposed answer: {candidate['answer']}\n"
            f"Proposed reasoning:\n{solution}"
        )
    user = (
        f"Question:\n{question}\n\n"
        + "\n\n".join(rendered)
        + "\n\nSolve the question independently, check each candidate's "
          "reasoning and arithmetic, and select the best supported proposed "
          "answer. Every candidate may be wrong. Return only "
          "'#### <selected number>'."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a strict mathematical verifier. Do not trust answer "
                "frequency. Select only from the proposed numeric answers."
            ),
        },
        {"role": "user", "content": user},
    ]


def verifier_select(model, tokenizer, question, answers, generations,
                    max_new_tokens=96):
    """Select a candidate with a greedy verification pass and safe fallback."""
    import torch

    candidates = candidate_rows(answers, generations)
    fallback = majority_vote(answers)
    if len(candidates) <= 1:
        return {
            "prediction": candidates[0]["answer"] if candidates else fallback,
            "raw_prediction": None,
            "generation": "",
            "used_fallback": False,
            "candidate_count": len(candidates),
        }

    inputs, _ = encode_chat(
        tokenizer, build_verifier_messages(question, candidates)
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generation = decode_completion(
        tokenizer, output[0], inputs.input_ids.shape[1]
    )
    raw_prediction = extract_answer(generation)
    allowed = {candidate["answer"] for candidate in candidates}
    accepted = raw_prediction in allowed
    return {
        "prediction": raw_prediction if accepted else fallback,
        "raw_prediction": raw_prediction,
        "generation": generation,
        "used_fallback": not accepted,
        "candidate_count": len(candidates),
    }


def artifact_paths(run_name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", run_name):
        raise ValueError("run-name must contain only lowercase letters, digits, ._-")
    RES.mkdir(parents=True, exist_ok=True)
    return {
        "checkpoint": RES / f"{run_name}_checkpoint.json",
        "raw": RES / f"{run_name}_traces.jsonl",
        "result": RES / f"{run_name}_results.json",
    }


def write_raw_records(records, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run(model, tokenizer, problems, args, paths, split_hash):
    """Run or resume one fully specified configuration."""
    import torch

    sizes = vote_sizes(args.k)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_name": args.run_name,
        "split": args.split,
        "split_file_sha256": split_hash,
        "model_key": args.model_key,
        "model_id": MODEL_SPECS[args.model_key]["id"],
        "prompt_style": args.prompt_style,
        "k": args.k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "verifier_max_new_tokens": args.verifier_max_new_tokens,
        "seed": args.seed,
        "device": args.resolved_device,
        "reported_selector": args.reported_selector,
        "frozen_config_sha256": args.frozen_config_sha256,
        "problem_ids": [problem["id"] for problem in problems],
    }
    records = []
    if paths["checkpoint"].exists():
        state = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
        expected = {key: state.get(key) for key in metadata}
        if expected != metadata or "records" not in state:
            raise RuntimeError("checkpoint configuration mismatch; use --fresh")
        records = state["records"]
        if [r.get("id") for r in records] != metadata["problem_ids"][:len(records)]:
            raise RuntimeError("checkpoint record order is invalid; use --fresh")
        write_raw_records(records, paths["raw"])
        print(f"[resume] {len(records)} complete records")

    started = time.time()
    for index, problem in enumerate(problems):
        if index < len(records):
            continue
        inputs, prompt = encode_chat(
            tokenizer, build_messages(problem["question"], args.prompt_style)
        )
        inputs = inputs.to(model.device)
        prompt_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        greedy_generation = decode_completion(
            tokenizer, output[0], prompt_length
        )
        greedy_prediction = extract_answer(greedy_generation)

        torch.manual_seed(args.seed + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + index)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.k,
                pad_token_id=tokenizer.eos_token_id,
            )
        generations = [
            decode_completion(tokenizer, output[j], prompt_length)
            for j in range(args.k)
        ]
        answers = [extract_answer(generation) for generation in generations]
        verifier = verifier_select(
            model, tokenizer, problem["question"], answers, generations,
            max_new_tokens=args.verifier_max_new_tokens,
        )
        verifier["correct"] = is_correct(
            verifier["prediction"], problem["gold"]
        )

        record = {
            "i": index,
            "id": problem["id"],
            "question": problem["question"],
            "gold": problem["gold"],
            "prompt": prompt,
            "greedy_generation": greedy_generation,
            "greedy_prediction": greedy_prediction,
            "greedy_correct": is_correct(greedy_prediction, problem["gold"]),
            "sample_generations": generations,
            "extracted": answers,
            "votes": {
                str(size): {
                    "prediction": majority_vote(answers[:size]),
                    "correct": is_correct(
                        majority_vote(answers[:size]), problem["gold"]
                    ),
                }
                for size in sizes
            },
            "verifier": verifier,
            "oracle_any_sample_correct": any(
                is_correct(answer, problem["gold"]) for answer in answers
            ),
        }
        records.append(record)
        atomic_json_dump({**metadata, "records": records}, paths["checkpoint"])
        write_raw_records(records, paths["raw"])

        elapsed = time.time() - started
        greedy_correct = sum(r["greedy_correct"] for r in records)
        majority_correct = sum(
            r["votes"][str(args.k)]["correct"] for r in records
        )
        verifier_correct = sum(r["verifier"]["correct"] for r in records)
        print(
            f"  [{index + 1}/{len(problems)}] "
            f"greedy={greedy_correct}/{len(records)} "
            f"vote={majority_correct}/{len(records)} "
            f"verify={verifier_correct}/{len(records)} "
            f"({elapsed:.0f}s)",
            flush=True,
        )
    return records, sizes, metadata


def summarize(records, sizes, metadata, args):
    n = len(records)
    greedy = [record["greedy_correct"] for record in records]
    votes = {
        size: [record["votes"][str(size)]["correct"] for record in records]
        for size in sizes
    }
    verified = [record["verifier"]["correct"] for record in records]
    greedy_count = sum(greedy)
    vote_counts = {size: sum(values) for size, values in votes.items()}
    verified_count = sum(verified)
    oracle_count = sum(r["oracle_any_sample_correct"] for r in records)
    valid_answers = sum(
        answer is not None
        for record in records
        for answer in record["extracted"]
    )
    if args.reported_selector == "majority":
        selected = votes[args.k]
        selected_count = vote_counts[args.k]
    else:
        selected = verified
        selected_count = verified_count
    return {
        **metadata,
        "num_threads": args.num_threads,
        "n_problems": n,
        "trace_records": n,
        "greedy_correct": greedy_count,
        "greedy_accuracy": 100 * greedy_count / n,
        "greedy_wilson_95": wilson_interval(greedy_count, n),
        "majority_correct": {str(k): vote_counts[k] for k in sizes},
        "majority_accuracy": {
            str(k): 100 * vote_counts[k] / n for k in sizes
        },
        "majority_wilson_95": {
            str(k): wilson_interval(vote_counts[k], n) for k in sizes
        },
        "verifier_correct": verified_count,
        "verifier_accuracy": 100 * verified_count / n,
        "verifier_wilson_95": wilson_interval(verified_count, n),
        "oracle_any_sample_correct": oracle_count,
        "oracle_any_sample_accuracy": 100 * oracle_count / n,
        "valid_sample_answers": valid_answers,
        "total_sample_answers": n * args.k,
        "mcnemar_exact_p_majority_vs_greedy": exact_mcnemar_p(
            greedy, votes[args.k]
        ),
        "mcnemar_exact_p_verifier_vs_greedy": exact_mcnemar_p(
            greedy, verified
        ),
        "selected_correct": selected_count,
        "selected_accuracy": 100 * selected_count / n,
        "selected_wilson_95": wilson_interval(selected_count, n),
        "mcnemar_exact_p_selected_vs_greedy": exact_mcnemar_p(
            greedy, selected
        ),
    }


def sha256_text(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--split", choices=("development", "evaluation"), required=True
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-key", choices=MODEL_SPECS, default="qwen0.5b")
    parser.add_argument(
        "--prompt-style", choices=PROMPT_STYLES, default="chat-check"
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="execution backend; auto uses CUDA when available",
    )
    parser.add_argument(
        "--reported-selector", choices=("majority", "verifier"),
        default="verifier",
    )
    parser.add_argument(
        "--frozen-config",
        help="required configuration manifest for evaluation runs",
    )
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.k < 1 or args.max_new_tokens < 1 or args.num_threads < 1:
        parser.error("k, max-new-tokens, and num-threads must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be positive and top-p in (0, 1]")

    paths = artifact_paths(args.run_name)

    problems, _ = load_locked_split(args.split, limit=args.limit)
    split_hash = sha256_text(SPLIT_PATH)
    args.frozen_config_sha256 = None
    if args.split == "evaluation":
        if args.limit is not None:
            parser.error("evaluation runs must use all 100 locked items")
        if not args.frozen_config:
            parser.error("evaluation runs require --frozen-config")
        frozen_path = pathlib.Path(args.frozen_config)
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        required = (
            "run_name", "model_key", "prompt_style", "k", "temperature",
            "top_p", "max_new_tokens", "verifier_max_new_tokens", "seed",
            "reported_selector",
        )
        mismatches = {
            key: (frozen.get(key), getattr(args, key))
            for key in required
            if frozen.get(key) != getattr(args, key)
        }
        if frozen.get("split_file_sha256") != split_hash:
            mismatches["split_file_sha256"] = (
                frozen.get("split_file_sha256"), split_hash
            )
        if mismatches:
            raise RuntimeError(f"frozen configuration mismatch: {mismatches}")
        args.frozen_config_sha256 = sha256_text(frozen_path)
    elif args.frozen_config:
        parser.error("--frozen-config is reserved for evaluation runs")
    model_spec = MODEL_SPECS[args.model_key]
    weight_files = list(model_spec["path"].glob("*.safetensors"))
    if not weight_files or sum(path.stat().st_size for path in weight_files) < 5e8:
        raise RuntimeError(
            f"complete local weights not found in {model_spec['path']}"
        )
    if args.fresh:
        for path in paths.values():
            if path.exists():
                path.unlink()


    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")
    args.resolved_device = args.device
    if args.device == "auto":
        args.resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[data] {args.split}: {len(problems)} locked GSM8K items")
    print(f"[model] {model_spec['id']} from {model_spec['path']}")
    print(
        f"[config] prompt={args.prompt_style}, temperature={args.temperature}, "
        f"k={args.k}, seed={args.seed}, device={args.resolved_device}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["path"], local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["path"], dtype="bfloat16", local_files_only=True
    )
    model.to(args.resolved_device)
    model.eval()

    records, sizes, metadata = run(
        model, tokenizer, problems, args, paths, split_hash
    )
    result = summarize(records, sizes, metadata, args)
    atomic_json_dump(result, paths["result"])
    print(json.dumps(result, indent=2))
    print(paths["result"])


if __name__ == "__main__":
    main()

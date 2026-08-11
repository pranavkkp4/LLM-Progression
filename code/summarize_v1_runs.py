"""Summarize and rank improved-v1 experiment artifacts."""
import argparse
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v1"


def candidate_rows(paths):
    rows = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        k = str(result["k"])
        common = {
            "run_name": result["run_name"],
            "split": result["split"],
            "n": result["n_problems"],
            "model_key": result["model_key"],
            "prompt_style": result["prompt_style"],
            "temperature": result["temperature"],
            "k": result["k"],
            "greedy_correct": result["greedy_correct"],
            "greedy_accuracy": result["greedy_accuracy"],
            "oracle_correct": result["oracle_any_sample_correct"],
            "oracle_accuracy": result["oracle_any_sample_accuracy"],
            "valid_answers": result["valid_sample_answers"],
            "total_answers": result["total_sample_answers"],
        }
        rows.append({
            **common,
            "selector": "majority",
            "selected_correct": result["majority_correct"][k],
            "selected_accuracy": result["majority_accuracy"][k],
            "mcnemar_p_vs_greedy": (
                result["mcnemar_exact_p_majority_vs_greedy"]
            ),
        })
        rows.append({
            **common,
            "selector": "verifier",
            "selected_correct": result["verifier_correct"],
            "selected_accuracy": result["verifier_accuracy"],
            "mcnemar_p_vs_greedy": (
                result["mcnemar_exact_p_verifier_vs_greedy"]
            ),
        })
    return rows


def ranking_key(row):
    """Higher accuracy first; ties prefer cheaper and more stable settings."""
    selector_cost = 0 if row["selector"] == "majority" else 1
    prompt_order = {
        "chat-concise": 0,
        "chat-check": 1,
        "chat-fewshot": 2,
    }
    return (
        -row["selected_accuracy"],
        -row["greedy_accuracy"],
        selector_cost,
        row["temperature"],
        prompt_order[row["prompt_style"]],
        row["run_name"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*_results.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    paths = sorted(RESULTS.glob(args.pattern))
    if not paths:
        raise RuntimeError(f"no result files match {args.pattern!r}")
    rows = sorted(candidate_rows(paths), key=ranking_key)
    fields = list(rows[0])
    if args.output:
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    widths = {
        field: max(len(field), *(len(str(row[field])) for row in rows))
        for field in (
            "run_name", "model_key", "prompt_style", "temperature",
            "selector", "greedy_correct", "selected_correct",
            "oracle_correct",
        )
    }
    display_fields = list(widths)
    print("  ".join(field.ljust(widths[field]) for field in display_fields))
    for row in rows:
        print(
            "  ".join(
                str(row[field]).ljust(widths[field])
                for field in display_fields
            )
        )


if __name__ == "__main__":
    main()

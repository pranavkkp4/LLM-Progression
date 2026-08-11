"""Freeze one development-selected configuration before evaluation."""
import argparse
import hashlib
import json
import pathlib

from exp4_self_consistency import atomic_json_dump

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "v1_frozen_config.json"
SPLIT_PATH = ROOT / "data" / "gsm8k_v1_splits.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("development_result")
    parser.add_argument(
        "--selector", choices=("majority", "verifier"), required=True
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    source_path = pathlib.Path(args.development_result)
    output_path = pathlib.Path(args.output)
    if output_path.exists():
        raise RuntimeError(f"refusing to replace frozen config: {output_path}")
    result = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        result["split"] != "development"
        or result["n_problems"] != 16
        or result["trace_records"] != 16
    ):
        raise ValueError("freeze only from a complete 16-item development run")
    split_manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split_sha = hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()
    if result["split_file_sha256"] != split_sha:
        raise ValueError("development result does not match split manifest hash")
    if result["problem_ids"] != split_manifest["development_ids"]:
        raise ValueError("development result does not use the locked IDs")

    fields = (
        "model_key", "model_id", "prompt_style", "k", "temperature", "top_p",
        "max_new_tokens", "verifier_max_new_tokens", "seed",
    )
    frozen = {
        "schema_version": 1,
        "run_name": "v1_final",
        "split": "evaluation",
        "split_file_sha256": result["split_file_sha256"],
        "reported_selector": args.selector,
        "selection_source": source_path.name,
        "selection_policy": (
            "Configuration chosen only from the locked 16-item development "
            "split. The 100 evaluation labels were not inspected before this "
            "manifest was written."
        ),
        **{field: result[field] for field in fields},
    }
    atomic_json_dump(frozen, output_path)
    print(json.dumps(frozen, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()

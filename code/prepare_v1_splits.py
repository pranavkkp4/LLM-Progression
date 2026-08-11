"""Lock disjoint GSM8K development and evaluation IDs before v1 tuning."""
import argparse
import hashlib
import json
import pathlib

import pandas as pd

from exp4_self_consistency import GSM8K_PARQUET, atomic_json_dump

ROOT = pathlib.Path(__file__).resolve().parent
ORIGINAL_CHECKPOINT = ROOT / "results" / "self_consistency_checkpoint.json"
SPLIT_PATH = ROOT / "data" / "gsm8k_v1_splits.json"
EVALUATION_SEED = 20260810


def parquet_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_splits():
    if not GSM8K_PARQUET.exists():
        raise FileNotFoundError(GSM8K_PARQUET)
    if not ORIGINAL_CHECKPOINT.exists():
        raise FileNotFoundError(ORIGINAL_CHECKPOINT)

    frame = pd.read_parquet(GSM8K_PARQUET)
    if len(frame) != 1319 or not {"question", "answer"} <= set(frame.columns):
        raise ValueError("expected the official 1,319-row GSM8K test split")

    checkpoint = json.loads(ORIGINAL_CHECKPOINT.read_text(encoding="utf-8"))
    development_ids = checkpoint["problem_ids"]
    if len(development_ids) != 16 or len(set(development_ids)) != 16:
        raise ValueError("the original development run must contain 16 unique IDs")

    development_indices = {
        int(identifier.removeprefix("gsm8k-test-"))
        for identifier in development_ids
    }
    if not development_indices <= set(frame.index):
        raise ValueError("development IDs do not match the committed parquet")

    remaining = frame.drop(index=list(development_indices))
    evaluation_indices = remaining.sample(
        n=100, random_state=EVALUATION_SEED
    ).index.tolist()
    evaluation_ids = [f"gsm8k-test-{index}" for index in evaluation_indices]
    if set(development_ids) & set(evaluation_ids):
        raise AssertionError("development and evaluation splits overlap")

    return {
        "schema_version": 1,
        "dataset": "openai/gsm8k test",
        "parquet_sha256": parquet_sha256(GSM8K_PARQUET),
        "policy": (
            "The original 16-item run is development data. The 100 evaluation "
            "IDs were sampled once from the remaining rows with pandas "
            f"random_state={EVALUATION_SEED} before v1 tuning."
        ),
        "development_ids": development_ids,
        "evaluation_ids": evaluation_ids,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing split file (not for the reported v1 study)",
    )
    args = parser.parse_args()
    if SPLIT_PATH.exists() and not args.force:
        raise RuntimeError(f"split file already exists: {SPLIT_PATH}")
    splits = build_splits()
    atomic_json_dump(splits, SPLIT_PATH)
    print(
        f"locked {len(splits['development_ids'])} development and "
        f"{len(splits['evaluation_ids'])} evaluation IDs"
    )
    print(f"parquet sha256={splits['parquet_sha256']}")
    print(SPLIT_PATH)


if __name__ == "__main__":
    main()

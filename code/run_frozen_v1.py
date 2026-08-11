"""Run the single frozen 100-item v1 evaluation configuration."""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "data" / "v1_frozen_config.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    config_path = pathlib.Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    command = [
        sys.executable,
        str(ROOT / "exp4_v1_improved.py"),
        "--run-name", config["run_name"],
        "--split", "evaluation",
        "--model-key", config["model_key"],
        "--prompt-style", config["prompt_style"],
        "--k", str(config["k"]),
        "--temperature", str(config["temperature"]),
        "--top-p", str(config["top_p"]),
        "--max-new-tokens", str(config["max_new_tokens"]),
        "--verifier-max-new-tokens",
        str(config["verifier_max_new_tokens"]),
        "--seed", str(config["seed"]),
        "--reported-selector", config["reported_selector"],
        "--frozen-config", str(config_path),
        "--num-threads", str(args.num_threads),
    ]
    if args.fresh:
        command.append("--fresh")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

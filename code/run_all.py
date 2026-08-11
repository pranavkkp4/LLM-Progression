"""Run the full analysis pipeline (exp1-exp3) and regenerate all outputs.

exp4 (the live LLM experiment) is intentionally separate: it needs the model
weights and takes ~1 hour on CPU. See README.md.
"""
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    for script in ["exp1_progression.py", "exp2_cost.py",
                   "exp3_specialization.py"]:
        print(f"\n===== {script} =====")
        subprocess.run([sys.executable, str(ROOT / script)], check=True)
    print("\nAll analysis experiments completed. Figures in ../paper/figures/, "
          "tables in ./results/.")
    print("To run the live experiment: python exp4_self_consistency.py "
          "(see README.md)")


if __name__ == "__main__":
    main()

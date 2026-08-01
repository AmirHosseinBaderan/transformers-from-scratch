"""
Main GPT pipeline entry point.

Runs the full pipeline in order:
  1. Data pipeline  -> python data_pipeline.py
  2. Train model    -> python -m mini_gpt.train
"""

import subprocess
import sys


def run_data_pipeline():
    """Step 1: Run the data pipeline to build vocabulary and encode datasets."""
    print("=" * 60)
    print("STEP 1: Running data pipeline")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "data_pipeline.py"],
    )

    if result.returncode != 0:
        print("Data pipeline failed with return code", result.returncode)
        sys.exit(result.returncode)

    print("Data pipeline completed successfully.\n")


def run_training():
    """Step 2: Train the MiniGPT model."""
    print("=" * 60)
    print("STEP 2: Running training")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "mini_gpt.train"],
    )

    if result.returncode != 0:
        print("Training failed with return code", result.returncode)
        sys.exit(result.returncode)

    print("Training completed successfully.\n")


def main():
    run_data_pipeline()
    run_training()

    print("=" * 60)
    print("Pipeline finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

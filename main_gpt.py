"""
Main GPT pipeline entry point.

Runs the full pipeline in order:
  1. Data pipeline  -> python data_pipeline.py
  2. Train model    -> python -m mini_gpt.train
"""

import subprocess
import sys

from common.utils.logger import logger


def run_data_pipeline():
    """Step 1: Run the data pipeline to build vocabulary and encode datasets."""
    logger.info("=" * 60)
    logger.info("STEP 1: Running data pipeline")
    logger.info("=" * 60)

    result = subprocess.run(
        [sys.executable, "data_pipeline.py"],
    )

    if result.returncode != 0:
        logger.error("Data pipeline failed with return code %s", result.returncode)
        sys.exit(result.returncode)

    logger.info("Data pipeline completed successfully.")


def run_training():
    """Step 2: Train the MiniGPT model."""
    logger.info("=" * 60)
    logger.info("STEP 2: Running training")
    logger.info("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "mini_gpt.train"],
    )

    if result.returncode != 0:
        logger.error("Training failed with return code %s", result.returncode)
        sys.exit(result.returncode)

    logger.info("Training completed successfully.")


def main():
    run_data_pipeline()
    run_training()

    logger.info("=" * 60)
    logger.info("Pipeline finished successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

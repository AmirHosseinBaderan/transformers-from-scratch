from pathlib import Path

import gc
import pandas as pd
import torch

from common.training.tensorboard_logger import TensorBoardLogger

from common.utils.logger import logger

from mini_t5.config import T5Config
from mini_t5.trainer import build_trainer


def main():
    logger.info("=" * 60)
    logger.info("MiniT5 Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Device: {T5Config.DEVICE}")
    logger.info(f"Batch Size: {T5Config.BATCH_SIZE}")
    logger.info(f"Gradient Accumulation Steps: {T5Config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Effective Batch Size: {T5Config.BATCH_SIZE * T5Config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Learning Rate: {T5Config.LEARNING_RATE}")
    logger.info(f"Epochs: {T5Config.EPOCHS}")
    logger.info(f"Mixed Precision: {T5Config.USE_MIXED_PRECISION}")
    logger.info(f"Gradient Checkpointing: {T5Config.USE_GRADIENT_CHECKPOINTING}")
    logger.info(f"Gradient Clip Norm: {T5Config.GRADIENT_CLIP_NORM}")
    logger.info(f"Early Stopping Patience: {T5Config.EARLY_STOPPING_PATIENCE}")
    logger.info(f"Checkpoint Dir: {T5Config.CHECKPOINT_DIR}")
    logger.info(f"TensorBoard Log Dir: {T5Config.TENSORBOARD_LOG_DIR}")
    logger.info("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    trainer = build_trainer()

    start_epoch = 0
    latest_checkpoint = trainer.checkpoint_manager.load_latest()
    if latest_checkpoint is not None:
        start_epoch = latest_checkpoint["epoch"]
        trainer.early_stopping.load_state_dict(
            latest_checkpoint.get(
                "early_stopping_state",
                trainer.early_stopping.state_dict(),
            )
        )
        logger.info(f"Resumed from epoch {start_epoch}")

    try:
        trainer.train(start_epoch=start_epoch)
        logger.info("Training complete!")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("Out of memory error! Try:")
            logger.error("  1. Reduce BATCH_SIZE in config.py")
            logger.error("  2. Increase GRADIENT_ACCUMULATION_STEPS")
            logger.error("  3. Enable USE_GRADIENT_CHECKPOINTING")
            logger.error("  4. Reduce NUM_LAYERS or EMBEDDING_DIM")
            logger.error("  5. Reduce MAX_LENGTH")
            logger.error("  6. Close other applications")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()

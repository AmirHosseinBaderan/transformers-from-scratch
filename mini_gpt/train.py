from pathlib import Path

import gc
import torch
from torch.utils.data import DataLoader

from mini_gpt.model import MiniGPT

from common.data.dataset import TextDataset

from common.training.losses import (
    LanguageModelLoss,
)

from common.training.checkpoint import (
    CheckpointManager,
)

from common.training.early_stopping import (
    EarlyStopping,
)

from common.training.tensorboard_logger import (
    TensorBoardLogger,
)

from common.utils.logger import logger

from common.configs.model_config import ModelConfig
from common.data.vocabulary import Vocabulary

from mini_gpt.trainer import Trainer


def build_trainer() -> Trainer:
    # Load datasets
    train_dataset = TextDataset(
        Path("common/data/processed/train.bin"),
        ModelConfig.BLOCK_SIZE,
    )

    val_dataset = TextDataset(
        Path("common/data/processed/validation.bin"),
        ModelConfig.BLOCK_SIZE,
    )

    # Configure DataLoader for small memory systems
    # pin_memory=True speeds up CPU->GPU transfer (only for CUDA)
    # num_workers parallelizes data loading
    # persistent_workers keeps workers alive between epochs
    train_loader = DataLoader(
        train_dataset,
        batch_size=ModelConfig.BATCH_SIZE,
        shuffle=True,
        pin_memory=ModelConfig.PIN_MEMORY and torch.cuda.is_available(),
        num_workers=ModelConfig.NUM_WORKERS,
        prefetch_factor=ModelConfig.PREFETCH_FACTOR if ModelConfig.NUM_WORKERS > 0 else None,
        persistent_workers=ModelConfig.NUM_WORKERS > 0,
        drop_last=True,  # Drop incomplete batches to avoid memory spikes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=ModelConfig.BATCH_SIZE,
        shuffle=False,
        pin_memory=ModelConfig.PIN_MEMORY and torch.cuda.is_available(),
        num_workers=ModelConfig.NUM_WORKERS,
        prefetch_factor=ModelConfig.PREFETCH_FACTOR if ModelConfig.NUM_WORKERS > 0 else None,
        persistent_workers=ModelConfig.NUM_WORKERS > 0,
        drop_last=True,
    )

    vocabulary = Vocabulary.load(ModelConfig.VOCAB_PATH)

    model = MiniGPT(
        vocab_size=vocabulary.size,
        block_size=ModelConfig.BLOCK_SIZE,
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        num_heads=ModelConfig.NUM_HEADS,
        num_layers=ModelConfig.NUM_LAYERS,
        dropout=ModelConfig.DROPOUT,
        use_gradient_checkpointing=ModelConfig.USE_GRADIENT_CHECKPOINTING,
    )

    model.to(ModelConfig.DEVICE)

    # Enable gradient checkpointing if configured
    if ModelConfig.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ModelConfig.LEARNING_RATE,
    )

    criterion = LanguageModelLoss()

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=ModelConfig.CHECKPOINT_DIR,
        model=model,
        optimizer=optimizer,
        keep_last_n=ModelConfig.KEEP_LAST_N_CHECKPOINTS,
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=ModelConfig.EARLY_STOPPING_PATIENCE,
        min_delta=ModelConfig.EARLY_STOPPING_MIN_DELTA,
        mode="min",
    )

    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger(ModelConfig.TENSORBOARD_LOG_DIR)

    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=ModelConfig.DEVICE,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
        tb_logger=tb_logger,
        epochs=ModelConfig.EPOCHS,
    )


def main():
    # Log memory optimization settings
    logger.info("=" * 60)
    logger.info("Memory-Optimized Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Device: {ModelConfig.DEVICE}")
    logger.info(f"Batch Size: {ModelConfig.BATCH_SIZE}")
    logger.info(f"Gradient Accumulation Steps: {ModelConfig.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Effective Batch Size: {ModelConfig.BATCH_SIZE * ModelConfig.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Mixed Precision: {ModelConfig.USE_MIXED_PRECISION}")
    logger.info(f"Gradient Checkpointing: {ModelConfig.USE_GRADIENT_CHECKPOINTING}")
    logger.info(f"Gradient Clip Norm: {ModelConfig.GRADIENT_CLIP_NORM}")
    logger.info(f"DataLoader Workers: {ModelConfig.NUM_WORKERS}")
    logger.info(f"Pin Memory: {ModelConfig.PIN_MEMORY}")
    logger.info("=" * 60)

    trainer = build_trainer()

    # Try to resume from latest checkpoint
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
            logger.error("  5. Reduce BLOCK_SIZE")
        raise
    finally:
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()

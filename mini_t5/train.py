from pathlib import Path

import gc
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim import AdamW

from common.training.checkpoint import CheckpointManager
from common.training.early_stopping import EarlyStopping
from common.training.tensorboard_logger import TensorBoardLogger

from common.utils.logger import logger

from config import T5Config
from model import MiniT5
from modules.dataset import TranslationDataset
from modules.tokenizer import CharacterTokenizer
from train_one_epoch import train_one_epoch
from validate_one_epoch import validate_one_epoch


def build_trainer():
    tokenizer = CharacterTokenizer()

    # Read all texts for vocabulary building
    train_df = pd.read_csv(T5Config.TRAIN_CSV_PATH)
    val_df = pd.read_csv(T5Config.VAL_CSV_PATH)

    texts = []
    texts.extend(train_df["source"].astype(str).tolist())
    texts.extend(train_df["target"].astype(str).tolist())
    texts.extend(val_df["source"].astype(str).tolist())
    texts.extend(val_df["target"].astype(str).tolist())

    tokenizer.fit(texts)

    train_dataset = TranslationDataset(
        T5Config.TRAIN_CSV_PATH,
        tokenizer,
        T5Config.MAX_LENGTH,
    )

    val_dataset = TranslationDataset(
        T5Config.VAL_CSV_PATH,
        tokenizer,
        T5Config.MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=T5Config.BATCH_SIZE,
        shuffle=True,
        pin_memory=T5Config.PIN_MEMORY and torch.cuda.is_available(),
        num_workers=T5Config.NUM_WORKERS,
        prefetch_factor=T5Config.PREFETCH_FACTOR,
        persistent_workers=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=T5Config.BATCH_SIZE,
        shuffle=False,
        pin_memory=T5Config.PIN_MEMORY and torch.cuda.is_available(),
        num_workers=T5Config.NUM_WORKERS,
        prefetch_factor=T5Config.PREFETCH_FACTOR,
        persistent_workers=False,
        drop_last=True,
    )

    model = MiniT5(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=T5Config.EMBEDDING_DIM,
        num_layers=T5Config.NUM_LAYERS,
        num_heads=T5Config.NUM_HEADS,
        ff_hidden_dim=T5Config.EMBEDDING_DIM * 4,
        max_length=T5Config.MAX_LENGTH,
        dropout=T5Config.DROPOUT,
    ).to(T5Config.DEVICE)

    if T5Config.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id
    )

    optimizer = AdamW(
        model.parameters(),
        lr=T5Config.LEARNING_RATE,
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=T5Config.CHECKPOINT_DIR,
        model=model,
        optimizer=optimizer,
        keep_last_n=T5Config.KEEP_LAST_N_CHECKPOINTS,
    )

    early_stopping = EarlyStopping(
        patience=T5Config.EARLY_STOPPING_PATIENCE,
        min_delta=T5Config.EARLY_STOPPING_MIN_DELTA,
        mode="min",
    )

    tb_logger = TensorBoardLogger(T5Config.TENSORBOARD_LOG_DIR)

    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "tokenizer": tokenizer,
        "checkpoint_manager": checkpoint_manager,
        "early_stopping": early_stopping,
        "tb_logger": tb_logger,
    }


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

    model = trainer["model"]
    optimizer = trainer["optimizer"]
    criterion = trainer["criterion"]
    train_loader = trainer["train_loader"]
    val_loader = trainer["val_loader"]
    tokenizer = trainer["tokenizer"]
    checkpoint_manager = trainer["checkpoint_manager"]
    early_stopping = trainer["early_stopping"]
    tb_logger = trainer["tb_logger"]

    # Resume from latest checkpoint if available
    start_epoch = 0
    latest_checkpoint = checkpoint_manager.load_latest()
    if latest_checkpoint is not None:
        start_epoch = latest_checkpoint["epoch"]
        early_stopping.load_state_dict(
            latest_checkpoint.get(
                "early_stopping_state",
                early_stopping.state_dict(),
            )
        )
        logger.info(f"Resumed from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, T5Config.EPOCHS):
            logger.info(f"Start epoch: {epoch + 1}")

            # Training
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=T5Config.DEVICE,
                tokenizer=tokenizer,
                gradient_accumulation_steps=T5Config.GRADIENT_ACCUMULATION_STEPS,
                use_mixed_precision=T5Config.USE_MIXED_PRECISION,
                gradient_clip_norm=T5Config.GRADIENT_CLIP_NORM,
            )

            # Aggressive cleanup after training before validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Validation
            val_loss = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=T5Config.DEVICE,
                tokenizer=tokenizer,
            )

            # Aggressive cleanup after validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Log metrics
            tb_logger.log_training_loss(train_loss, epoch)
            tb_logger.log_validation_loss(val_loss, epoch)

            # Log learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            tb_logger.log_learning_rate(current_lr, epoch)

            # Check if this is the best model
            is_best = val_loss < checkpoint_manager.best_val_loss

            # Sync early stopping state to checkpoint manager
            checkpoint_manager.early_stopping_state = (
                early_stopping.state_dict()
            )

            # Save checkpoint
            checkpoint_manager.save(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                is_best=is_best,
            )

            # Early stopping check
            if early_stopping(val_loss):
                logger.info("Early stopping triggered!")
                break

            # Extra cleanup between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

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
        # Clean up loaders to free file mappings
        del train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        tb_logger.close()


if __name__ == "__main__":
    main()

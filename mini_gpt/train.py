from pathlib import Path

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

from mini_gpt.config import GPTConfig
from common.data.vocabulary import Vocabulary
from common.configs.model_config import ModelConfig

from mini_gpt.train_one_epoch import train_one_epoch
from mini_gpt.validate_one_epoch import validate_one_epoch


def main():
    # Load datasets
    train_dataset = TextDataset(
        Path("common/data/processed/train.bin"),
        GPTConfig.BLOCK_SIZE,
    )

    val_dataset = TextDataset(
        Path("common/data/processed/validation.bin"),
        GPTConfig.BLOCK_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=GPTConfig.BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=GPTConfig.BATCH_SIZE,
        shuffle=False,
    )

    vocabulary = Vocabulary.load(ModelConfig.VOCAB_PATH)

    model = MiniGPT(
        vocab_size=vocabulary.size,
        block_size=GPTConfig.BLOCK_SIZE,
        embedding_dim=GPTConfig.EMBEDDING_DIM,
        num_heads=GPTConfig.NUM_HEADS,
        num_layers=GPTConfig.NUM_LAYERS,
        dropout=GPTConfig.DROPOUT,
    )

    model.to(GPTConfig.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GPTConfig.LEARNING_RATE,
    )

    criterion = LanguageModelLoss()

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=GPTConfig.CHECKPOINT_DIR,
        model=model,
        optimizer=optimizer,
        keep_last_n=GPTConfig.KEEP_LAST_N_CHECKPOINTS,
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=GPTConfig.EARLY_STOPPING_PATIENCE,
        min_delta=GPTConfig.EARLY_STOPPING_MIN_DELTA,
        mode="min",
    )

    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger(GPTConfig.TENSORBOARD_LOG_DIR)

    # Try to resume from latest checkpoint
    start_epoch = 0
    latest_checkpoint = checkpoint_manager.load_latest()
    if latest_checkpoint is not None:
        start_epoch = latest_checkpoint["epoch"]
        early_stopping.load_state_dict(
            latest_checkpoint.get("early_stopping_state", early_stopping.state_dict())
        )
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, GPTConfig.EPOCHS):
        # Training
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            GPTConfig.DEVICE,
        )

        # Validation
        val_loss = validate_one_epoch(
            model,
            val_loader,
            criterion,
            GPTConfig.DEVICE,
        )

        # Log metrics
        tb_logger.log_training_loss(train_loss, epoch)
        tb_logger.log_validation_loss(val_loss, epoch)

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        tb_logger.log_learning_rate(current_lr, epoch)

        logger.info(
            f"Epoch {epoch+1}/{GPTConfig.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Check if this is the best model
        is_best = val_loss < checkpoint_manager.best_val_loss

        # Sync early stopping state to checkpoint manager
        checkpoint_manager.early_stopping_state = early_stopping.state_dict()

        # Save checkpoint
        checkpoint_manager.save(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            is_best=is_best,
        )

        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    tb_logger.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

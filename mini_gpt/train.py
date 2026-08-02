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

from mini_gpt.trainer import Trainer


def build_trainer() -> Trainer:
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

    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=GPTConfig.DEVICE,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
        tb_logger=tb_logger,
        epochs=GPTConfig.EPOCHS,
    )


def main():
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

    trainer.train(start_epoch=start_epoch)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

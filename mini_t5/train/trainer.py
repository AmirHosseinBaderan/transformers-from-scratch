from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.training.checkpoint import CheckpointManager
from common.training.early_stopping import EarlyStopping
from common.training.tensorboard_logger import TensorBoardLogger
from common.utils.logger import logger

from mini_t5.config import T5Config
from mini_t5.model import MiniT5
from mini_t5.modules.dataset import TranslationDataset
from mini_t5.modules.tokenizer import CharacterTokenizer
from mini_t5.train_one_epoch import train_one_epoch
from mini_t5.validate_one_epoch import validate_one_epoch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mini_t5.model import MiniT5
    from mini_t5.modules.tokenizer import CharacterTokenizer


def build_trainer():
    tokenizer = CharacterTokenizer()

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
        use_gradient_checkpointing=T5Config.USE_GRADIENT_CHECKPOINTING,
    ).to(T5Config.DEVICE)

    if T5Config.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    criterion = torch.nn.CrossEntropyLoss(
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

    return T5Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=T5Config.DEVICE,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
        tb_logger=tb_logger,
        epochs=T5Config.EPOCHS,
    )


class T5Trainer:
    """
    Orchestrates the training loop for MiniT5 with ultra low-memory optimizations.
    """

    def __init__(
        self,
        model: MiniT5,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: CharacterTokenizer,
        device: str,
        checkpoint_manager: CheckpointManager,
        early_stopping: EarlyStopping,
        tb_logger: TensorBoardLogger,
        epochs: int,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping
        self.tb_logger = tb_logger
        self.epochs = epochs

    def _cleanup_memory(self, clear_cuda: bool = True) -> None:
        if clear_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for _ in range(3):
            gc.collect()

    def train(self, start_epoch: int = 0) -> None:
        for epoch in range(start_epoch, self.epochs):
            logger.info(f"Start epoch: {epoch + 1}")

            train_loss = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                tokenizer=self.tokenizer,
                gradient_accumulation_steps=T5Config.GRADIENT_ACCUMULATION_STEPS,
                use_mixed_precision=T5Config.USE_MIXED_PRECISION,
                gradient_clip_norm=T5Config.GRADIENT_CLIP_NORM,
            )

            self._cleanup_memory()

            val_loss = validate_one_epoch(
                model=self.model,
                loader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                tokenizer=self.tokenizer,
            )

            self._cleanup_memory()

            self.tb_logger.log_training_loss(train_loss, epoch)
            self.tb_logger.log_validation_loss(val_loss, epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.tb_logger.log_learning_rate(current_lr, epoch)

            is_best = val_loss < self.checkpoint_manager.best_val_loss

            self.checkpoint_manager.early_stopping_state = (
                self.early_stopping.state_dict()
            )

            self.checkpoint_manager.save(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                is_best=is_best,
            )

            if self.early_stopping(val_loss):
                logger.info("Early stopping triggered!")
                break

            self._cleanup_memory()

        del self.train_loader
        del self.val_loader
        self._cleanup_memory()

        self.tb_logger.close()

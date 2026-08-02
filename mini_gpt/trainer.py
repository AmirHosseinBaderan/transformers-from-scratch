from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch

from common.training.checkpoint import CheckpointManager
from common.training.early_stopping import EarlyStopping
from common.training.tensorboard_logger import TensorBoardLogger
from common.training.losses import LanguageModelLoss
from common.configs.model_config import ModelConfig
from common.utils.logger import logger

from mini_gpt.train_one_epoch import train_one_epoch
from mini_gpt.validate_one_epoch import validate_one_epoch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mini_gpt.model import MiniGPT


class Trainer:
    """
    Orchestrates the training loop for MiniGPT with ultra low-memory optimizations.
    """

    def __init__(
        self,
        model: MiniGPT,
        optimizer: torch.optim.Optimizer,
        criterion: LanguageModelLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
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
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping
        self.tb_logger = tb_logger
        self.epochs = epochs

    def _cleanup_memory(self, clear_cuda: bool = True) -> None:
        """
        Aggressive memory cleanup.
        """
        # Clear CUDA cache
        if clear_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Run garbage collection multiple times
        for _ in range(3):
            gc.collect()

    def train(self, start_epoch: int = 0) -> None:
        """
        Run the training loop with ultra low-memory optimizations.

        Args:
            start_epoch: Epoch to start from (for resuming).
        """
        for epoch in range(start_epoch, self.epochs):
            logger.info(f"Start epoch : {epoch + 1}")

            # Training
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                gradient_accumulation_steps=ModelConfig.GRADIENT_ACCUMULATION_STEPS,
                use_mixed_precision=ModelConfig.USE_MIXED_PRECISION,
                gradient_clip_norm=ModelConfig.GRADIENT_CLIP_NORM,
            )

            # Aggressive cleanup after training before validation
            self._cleanup_memory()

            # Validation
            val_loss = validate_one_epoch(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
            )

            # Aggressive cleanup after validation
            self._cleanup_memory()

            # Log metrics
            self.tb_logger.log_training_loss(train_loss, epoch)
            self.tb_logger.log_validation_loss(val_loss, epoch)

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.tb_logger.log_learning_rate(current_lr, epoch)

            # Check if this is the best model
            is_best = val_loss < self.checkpoint_manager.best_val_loss

            # Sync early stopping state to checkpoint manager
            self.checkpoint_manager.early_stopping_state = (
                self.early_stopping.state_dict()
            )

            # Save checkpoint
            self.checkpoint_manager.save(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                is_best=is_best,
            )

            # Early stopping check
            if self.early_stopping(val_loss):
                break

            # Extra cleanup between epochs
            self._cleanup_memory()

        # Clean up loaders to free file mappings
        del self.train_loader
        del self.val_loader
        self._cleanup_memory()

        self.tb_logger.close()

from __future__ import annotations

from typing import TYPE_CHECKING

from common.training.checkpoint import CheckpointManager
from common.training.early_stopping import EarlyStopping
from common.training.tensorboard_logger import TensorBoardLogger
from common.training.losses import LanguageModelLoss

from mini_gpt.train_one_epoch import train_one_epoch
from mini_gpt.validate_one_epoch import validate_one_epoch

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
    from mini_gpt.model import MiniGPT


class Trainer:
    """
    Orchestrates the training loop for MiniGPT.
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

    def train(self, start_epoch: int = 0) -> None:
        """
        Run the training loop.

        Args:
            start_epoch: Epoch to start from (for resuming).
        """
        for epoch in range(start_epoch, self.epochs):
            # Training
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
            )

            # Validation
            val_loss = validate_one_epoch(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
            )

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

        self.tb_logger.close()

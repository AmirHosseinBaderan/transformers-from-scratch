from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from common.utils.logger import logger


class TensorBoardLogger:
    """
    Logger for TensorBoard that tracks training and validation metrics.
    """

    def __init__(self, log_dir: str | Path):
        """
        Args:
            log_dir: Directory to save TensorBoard logs.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        logger.info(f"TensorBoard logs will be saved to: {self.log_dir}")

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
    ) -> None:
        """
        Log a scalar value.

        Args:
            tag: Name of the scalar.
            value: Scalar value to log.
            step: Global step value.
        """
        self.writer.add_scalar(tag, value, step)

    def log_training_loss(
        self,
        loss: float,
        epoch: int,
    ) -> None:
        """
        Log training loss for an epoch.

        Args:
            loss: Training loss value.
            epoch: Current epoch number.
        """
        self.log_scalar("Loss/train", loss, epoch)

    def log_validation_loss(
        self,
        loss: float,
        epoch: int,
    ) -> None:
        """
        Log validation loss for an epoch.

        Args:
            loss: Validation loss value.
            epoch: Current epoch number.
        """
        self.log_scalar("Loss/val", loss, epoch)

    def log_learning_rate(
        self,
        lr: float,
        epoch: int,
    ) -> None:
        """
        Log learning rate for an epoch.

        Args:
            lr: Learning rate value.
            epoch: Current epoch number.
        """
        self.log_scalar("Learning_Rate", lr, epoch)

    def close(self) -> None:
        """
        Close the TensorBoard writer.
        """
        self.writer.close()
        logger.info("TensorBoard writer closed.")

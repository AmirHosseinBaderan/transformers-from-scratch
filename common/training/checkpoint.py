from __future__ import annotations

from pathlib import Path

import torch

from common.utils.logger import logger


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device of the first parameter of the model.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class CheckpointManager:
    """
    Manages saving and loading model checkpoints during training.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        keep_last_n: int = 3,
        early_stopping_state: dict | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.keep_last_n = keep_last_n
        self.best_val_loss = float("inf")
        self.early_stopping_state = early_stopping_state or {}

    def save(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        is_best: bool = False,
    ) -> None:
        """
        Save a checkpoint.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch (optional).
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "early_stopping_state": self.early_stopping_state,
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model if this is the best validation loss
        if is_best and val_loss is not None:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path} (val_loss={val_loss:.4f})")
            self.best_val_loss = val_loss

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def load(self, checkpoint_path: str | Path) -> dict:
        """
        Load a checkpoint from disk.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            Dictionary containing checkpoint data.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=_get_model_device(self.model),
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "val_loss" in checkpoint and checkpoint["val_loss"] is not None:
            self.best_val_loss = checkpoint["val_loss"]

        if "early_stopping_state" in checkpoint:
            self.early_stopping_state = checkpoint["early_stopping_state"]

        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch={checkpoint['epoch']}, train_loss={checkpoint['train_loss']:.4f})"
        )

        return checkpoint

    def load_latest(self) -> dict | None:
        """
        Load the latest checkpoint from the checkpoint directory.

        Returns:
            Checkpoint dictionary or None if no checkpoints exist.
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        if not checkpoints:
            return None

        return self.load(checkpoints[-1])

    def _cleanup_old_checkpoints(self) -> None:
        """
        Remove old checkpoints, keeping only the last `keep_last_n` checkpoints.
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        while len(checkpoints) > self.keep_last_n:
            old_checkpoint = checkpoints.pop(0)
            old_checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {old_checkpoint}")

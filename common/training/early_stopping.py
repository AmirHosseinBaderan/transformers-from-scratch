from __future__ import annotations

from common.utils.logger import logger


class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric stops improving.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should be stopped.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            return False

        if self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter}/{self.patience}"
            )

        if self.counter >= self.patience:
            self.early_stop = True
            logger.info("Early stopping triggered!")
            return True

        return False

    def _is_better(self, score: float) -> bool:
        """
        Check if the current score is better than the best score.
        """
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def state_dict(self) -> dict:
        """
        Return the state of the early stopping object.
        """
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state of the early stopping object.
        """
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.early_stop = state_dict["early_stop"]

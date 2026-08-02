from __future__ import annotations

from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np
import torch

from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    pass


class TextDataset(Dataset):

    def __init__(
            self,
            path: Path,
            block_size: int,
    ):

        self._tokens = np.memmap(
            path,
            dtype=np.uint16,
            mode="r",
        )

        self._block_size = block_size

        # Pre-allocate buffers for efficient tensor creation
        # This avoids creating new numpy arrays on every __getitem__ call
        self._x_buffer = np.empty(
            block_size,
            dtype=np.int64,
        )
        self._y_buffer = np.empty(
            block_size,
            dtype=np.int64,
        )

        if len(self._tokens) <= block_size:
            raise ValueError(
                "Dataset is smaller than block size"
            )


    def __len__(self):

        return len(self._tokens) - self._block_size


    def __getitem__(
            self,
            index: int,
    ) -> tuple[Tensor, Tensor]:

        # Copy data into pre-allocated buffers (avoids new allocation)
        self._x_buffer[:] = self._tokens[index:index + self._block_size]
        self._y_buffer[:] = self._tokens[index + 1:index + self._block_size + 1]

        # Use from_numpy to create tensors that share memory (zero-copy)
        x_tensor = torch.from_numpy(self._x_buffer.copy())
        y_tensor = torch.from_numpy(self._y_buffer.copy())

        return (
            x_tensor,
            y_tensor,
        )

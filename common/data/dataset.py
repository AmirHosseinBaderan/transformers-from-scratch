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

        # Slice memmap and convert directly to tensors
        # torch.tensor creates a copy, which is necessary since memmap is shared
        x = torch.tensor(
            self._tokens[index:index + self._block_size],
            dtype=torch.long,
        )
        y = torch.tensor(
            self._tokens[index + 1:index + self._block_size + 1],
            dtype=torch.long,
        )

        return x, y

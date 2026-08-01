from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from torch import Tensor
from torch.utils.data import Dataset


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

        x = self._tokens[
            index:index + self._block_size
        ]

        y = self._tokens[
            index + 1:index + self._block_size + 1
        ]


        return (
            torch.tensor(
                x,
                dtype=torch.long,
            ),
            torch.tensor(
                y,
                dtype=torch.long,
            ),
        )
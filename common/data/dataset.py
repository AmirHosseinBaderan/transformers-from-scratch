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
        stride: int | None = None,
    ):

        self._tokens = np.memmap(
            path,
            dtype=np.uint16,
            mode="r",
        )

        self._block_size = block_size

        self._stride = (
            stride
            if stride is not None
            else block_size
        )

        if len(self._tokens) <= block_size:
            raise ValueError(
                "Dataset is smaller than block size"
            )


    def __len__(self):

        return (
            len(self._tokens) - self._block_size
        ) // self._stride


    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor]:

        start = index * self._stride

        x = torch.tensor(
            self._tokens[
                start:
                start + self._block_size
            ],
            dtype=torch.long,
        )

        y = torch.tensor(
            self._tokens[
                start + 1:
                start + self._block_size + 1
            ],
            dtype=torch.long,
        )

        return x, y
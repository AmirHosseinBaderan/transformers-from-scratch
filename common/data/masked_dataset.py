from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset

from common.data.preprocessing.masker import Masker


class MaskedDataset(Dataset):

    def __init__(
        self,
        path: Path,
        block_size: int,
        masker: Masker,
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

        self._masker = masker

        if len(self._tokens) <= block_size:
            raise ValueError(
                "Dataset is smaller than block size."
            )

    def __len__(self):

        return (
            len(self._tokens) - self._block_size
        ) // self._stride

    def __getitem__(
        self,
        index: int,
    ):

        start = index * self._stride

        token_ids = self._tokens[
            start:
            start + self._block_size
        ].tolist()

        input_ids, labels = self._masker.mask(
            token_ids
        )

        attention_mask = torch.ones(
            self._block_size,
            dtype=torch.long,
        )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
            "attention_mask": attention_mask,
        }
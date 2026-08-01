from __future__ import annotations

from torch import Tensor
from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    """
    Dataset for next-token prediction.
    """

    def __init__(
        self,
        token_ids: list[int],
        block_size: int,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero.")

        if len(token_ids) <= block_size:
            raise ValueError(
                "token_ids must contain more elements than block_size."
            )

        self._token_ids = token_ids
        self._block_size = block_size

    def __len__(self) -> int:
        return len(self._token_ids) - self._block_size

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor]:
        start = index
        end = start + self._block_size

        input_ids = self._token_ids[start:end]
        target_ids = self._token_ids[start + 1:end + 1]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )
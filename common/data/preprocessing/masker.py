from __future__ import annotations

import random

from common.data.vocabulary import Vocabulary


class Masker:

    def __init__(
        self,
        vocabulary: Vocabulary,
        mask_probability: float = 0.15,
    ):

        self._vocabulary = vocabulary
        self._mask_probability = mask_probability

        self._mask_id = vocabulary.token_to_id("<MASK>")

    def mask(
        self,
        token_ids: list[int],
    ) -> tuple[list[int], list[int]]:

        input_ids = token_ids.copy()

        labels = [-100] * len(token_ids)

        for index, token_id in enumerate(token_ids):

            if random.random() >= self._mask_probability:
                continue

            input_ids[index] = self._mask_id
            labels[index] = token_id

        return input_ids, labels
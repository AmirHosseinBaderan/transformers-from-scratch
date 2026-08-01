from __future__ import annotations

from typing import Iterable

from common.data.vocabulary import Vocabulary


class CharacterTokenizer:

    def __init__(
            self,
            vocabulary: Vocabulary,
    ):
        self._vocabulary = vocabulary


    def encode(
            self,
            text: str,
    ) -> list[int]:

        return [
            self._vocabulary.token_to_id(char)
            for char in text
        ]


    def encode_iterable(
            self,
            texts: Iterable[str],
    ):

        for text in texts:
            for char in text:
                yield self._vocabulary.token_to_id(char)


    def decode(
            self,
            ids: list[int],
    ) -> str:

        return "".join(
            self._vocabulary.id_to_token(i)
            for i in ids
        )
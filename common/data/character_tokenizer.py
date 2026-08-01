from __future__ import annotations

from typing import Iterable, Iterator

from common.data.vocabulary import Vocabulary


class CharacterTokenizer:
    """
    Character-level tokenizer.

    Encodes text into token ids and decodes token ids back into text.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        eos_token: str = "<EOS>",
    ):
        self._vocabulary = vocabulary
        self._eos_id = (
            vocabulary.token_to_id(eos_token)
            if eos_token in vocabulary
            else None
        )

    def encode(
        self,
        text: str,
    ) -> list[int]:
        token_to_id = self._vocabulary.token_to_id

        return [
            token_to_id(char)
            for char in text
        ]

    def encode_iterable(
        self,
        texts: Iterable[str],
        add_eos: bool = True,
    ) -> Iterator[int]:

        token_to_id = self._vocabulary.token_to_id
        eos_id = self._eos_id

        for text in texts:

            for char in text:
                yield token_to_id(char)

            if add_eos and eos_id is not None:
                yield eos_id

    def decode(
        self,
        token_ids: Iterable[int],
    ) -> str:

        return "".join(
            self._vocabulary.id_to_token(token_id)
            for token_id in token_ids
        )
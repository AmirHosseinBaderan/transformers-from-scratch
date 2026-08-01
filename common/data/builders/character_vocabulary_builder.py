from __future__ import annotations

from collections import Counter
from typing import Iterable

from common.data.vocabulary import Vocabulary


class CharacterVocabularyBuilder:

    def __init__(
            self,
            special_tokens: list[str] | None = None,
    ):
        self._special_tokens = special_tokens or [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
        ]


    def build(
            self,
            texts: Iterable[str],
    ) -> Vocabulary:

        counter = Counter()

        for text in texts:
            counter.update(text)


        vocabulary = Vocabulary(
            special_tokens=self._special_tokens
        )


        for char, _ in counter.most_common():
            vocabulary.add_token(char)


        return vocabulary
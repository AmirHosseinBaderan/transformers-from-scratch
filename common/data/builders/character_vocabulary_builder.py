from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from common.data.vocabulary import Vocabulary


class CharacterVocabularyBuilder:

    DEFAULT_SPECIAL_TOKENS = (
        "<PAD>",
        "<UNK>",
        "<CLS>",
        "<SEP>",
        "<MASK>",
    )

    def __init__(
        self,
        special_tokens: Iterable[str] | None = None,
    ):

        self._special_tokens = list(
            special_tokens
            if special_tokens is not None
            else self.DEFAULT_SPECIAL_TOKENS
        )

    def build(
        self,
        texts: Iterable[str],
    ) -> Vocabulary:

        counter = Counter()

        for text in texts:
            counter.update(text)

        vocabulary = Vocabulary(
            special_tokens=self._special_tokens,
        )

        for character, _ in counter.most_common():
            vocabulary.add_token(character)

        return vocabulary
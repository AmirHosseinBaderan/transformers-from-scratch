from __future__ import annotations

from common.data.vocabulary import Vocabulary
from common.data.builders.vocabulary_builder import VocabularyBuilder


class CharacterVocabularyBuilder(VocabularyBuilder):
    """
    Builds a character-level vocabulary from raw text.
    """

    def __init__(
        self,
        special_tokens: list[str] | None = None,
    ) -> None:
        self._special_tokens = special_tokens or []

    def build(
        self,
        text: str,
    ) -> Vocabulary:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        vocabulary = Vocabulary(
            special_tokens=self._special_tokens,
        )

        characters = sorted(set(text))

        vocabulary.add_tokens(characters)

        return vocabulary
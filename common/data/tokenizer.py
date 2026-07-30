from __future__ import annotations

from abc import ABC, abstractmethod

from common.data.vocabulary import Vocabulary


class Tokenizer(ABC):
    """
    Base interface for all tokenizers.

    Every tokenizer is responsible for converting text into token IDs
    and converting token IDs back into text.
    """

    @property
    @abstractmethod
    def vocabulary(self) -> Vocabulary:
        """
        Returns the tokenizer vocabulary.
        """
        ...

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size.
        """
        return len(self.vocabulary)

    @abstractmethod
    def encode(
        self,
        text: str,
    ) -> list[int]:
        """
        Converts text into token IDs.
        """
        ...

    @abstractmethod
    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        """
        Converts token IDs back into text.
        """
        ...
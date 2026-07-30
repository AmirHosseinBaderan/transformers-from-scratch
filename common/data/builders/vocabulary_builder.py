from __future__ import annotations

from abc import ABC, abstractmethod

from common.data.vocabulary import Vocabulary


class VocabularyBuilder(ABC):
    """
    Base interface for vocabulary builders.
    """

    @abstractmethod
    def build(
        self,
        text: str,
    ) -> Vocabulary:
        """
        Build a vocabulary from raw text.
        """
        ...
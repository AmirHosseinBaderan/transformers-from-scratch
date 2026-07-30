from __future__ import annotations

from common.data.tokenizer import Tokenizer
from common.data.vocabulary import Vocabulary

class CharacterTokenizer(Tokenizer):
    """
    Character-Level Tokenizer
    """

    def __init__(self, vocabulary: Vocabulary):
        self._vocabulary = vocabulary

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def encode(self,text:str)-> list[int]:
        tokens = list(text)
        return self._vocabulary.encode(tokens)

    def decode(self,token_ids:list[int])-> str:
        tokens = self._vocabulary.decode(token_ids)
        return "".join(tokens)
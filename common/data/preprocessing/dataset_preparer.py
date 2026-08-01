from __future__ import annotations

from array import array
from pathlib import Path
from typing import Iterable

from common.data.builders.character_vocabulary_builder import (
    CharacterVocabularyBuilder,
)
from common.data.character_tokenizer import CharacterTokenizer
from common.data.vocabulary import Vocabulary


class DatasetPreparer:
    """
    Prepares datasets for language model training.

    Responsibilities:

    - Build vocabulary
    - Save vocabulary
    - Encode text into binary token files
    """

    def __init__(
        self,
        vocabulary_builder: CharacterVocabularyBuilder,
    ):
        self._builder = vocabulary_builder

    def prepare_vocabulary(
        self,
        texts: Iterable[str],
        output_path: Path,
    ) -> Vocabulary:

        vocabulary = self._builder.build(texts)

        vocabulary.save(output_path)

        return vocabulary

    def encode_to_file(
        self,
        texts: Iterable[str],
        tokenizer: CharacterTokenizer,
        output_path: Path,
        buffer_size: int = 65536,
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # unsigned int (32-bit)
        buffer = array("I")

        with output_path.open("wb") as file:

            for token in tokenizer.encode_iterable(texts):

                buffer.append(token)

                if len(buffer) >= buffer_size:
                    buffer.tofile(file)
                    buffer.clear()

            if buffer:
                buffer.tofile(file)
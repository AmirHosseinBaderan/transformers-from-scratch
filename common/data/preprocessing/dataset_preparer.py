from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from common.data.builders.character_vocabulary_builder import (
    CharacterVocabularyBuilder,
)

from common.data.character_tokenizer import CharacterTokenizer
from common.data.vocabulary import Vocabulary


class DatasetPreparer:

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


        vocabulary = self._builder.build(
            texts
        )

        vocabulary.save(
            output_path
        )

        return vocabulary


    def encode_to_file(
            self,
            texts: Iterable[str],
            tokenizer: CharacterTokenizer,
            output_path: Path,
    ) -> None:


        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        with output_path.open("wb") as file:

            for token in tokenizer.encode_iterable(texts):

                np.asarray(
                    [token],
                    dtype=np.uint16,
                ).tofile(file)
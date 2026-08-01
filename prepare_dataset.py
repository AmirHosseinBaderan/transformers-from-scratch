from pathlib import Path

from common.data.readers.csv_reader import CSVReader
from common.data.builders.character_vocabulary_builder import (
    CharacterVocabularyBuilder,
)

from common.data.character_tokenizer import CharacterTokenizer

from common.data.preprocessing.dataset_preparer import (
    DatasetPreparer,
)


RAW = Path("common/data/raw")
OUTPUT = Path("common/data/processed")


def main():

    train_path = RAW / "train.csv"
    validation_path = RAW / "validation.csv"


    preparer = DatasetPreparer(
        CharacterVocabularyBuilder()
    )


    print("Building vocabulary...")


    vocabulary = preparer.prepare_vocabulary(
        CSVReader(train_path).read(),
        OUTPUT / "vocab.json",
    )


    tokenizer = CharacterTokenizer(
        vocabulary
    )


    print("Encoding train...")


    preparer.encode_to_file(
        CSVReader(train_path).read(),
        tokenizer,
        OUTPUT / "train.bin",
    )


    print("Encoding validation...")


    preparer.encode_to_file(
        CSVReader(validation_path).read(),
        tokenizer,
        OUTPUT / "validation.bin",
    )


    print("Done")
    print(
        "Vocabulary size:",
        len(vocabulary)
    )


if __name__ == "__main__":
    main()
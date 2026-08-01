from pathlib import Path

from common.data.builders.character_vocabulary_builder import (
    CharacterVocabularyBuilder,
)

from common.data.character_tokenizer import CharacterTokenizer
from common.data.readers.csv_reader import CSVReader
from common.data.preprocessing.dataset_preparer import DatasetPreparer


RAW_DATA_DIR = Path(
    "common/data/raw"
)

PROCESSED_DATA_DIR = Path(
    "common/data/processed"
)


TRAIN_FILE = RAW_DATA_DIR / "train.csv"
VALIDATION_FILE = RAW_DATA_DIR / "validation.csv"


VOCAB_FILE = PROCESSED_DATA_DIR / "vocab.json"
TRAIN_BIN = PROCESSED_DATA_DIR / "train.bin"
VALIDATION_BIN = PROCESSED_DATA_DIR / "validation.bin"


def main():

    print("Creating dataset preparer...")

    preparer = DatasetPreparer(
        CharacterVocabularyBuilder()
    )


    print("Building vocabulary...")


    train_reader = CSVReader(
        TRAIN_FILE
    )


    vocabulary = preparer.prepare_vocabulary(
        train_reader.read(),
        VOCAB_FILE,
    )


    print(
        "Vocabulary size:",
        len(vocabulary)
    )


    tokenizer = CharacterTokenizer(
        vocabulary
    )


    print("Encoding train dataset...")


    train_reader = CSVReader(
        TRAIN_FILE
    )


    preparer.encode_to_file(
        train_reader.read(),
        tokenizer,
        TRAIN_BIN,
    )


    print("Encoding validation dataset...")


    validation_reader = CSVReader(
        VALIDATION_FILE
    )


    preparer.encode_to_file(
        validation_reader.read(),
        tokenizer,
        VALIDATION_BIN,
    )


    print("Dataset preparation completed.")

    print()

    print("Generated files:")

    print(
        VOCAB_FILE
    )

    print(
        TRAIN_BIN
    )

    print(
        VALIDATION_BIN
    )


if __name__ == "__main__":
    main()
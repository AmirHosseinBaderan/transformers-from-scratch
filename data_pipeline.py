import csv
from pathlib import Path

from tqdm import tqdm

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


def count_csv_rows(path: Path) -> int:
    """
    Count the number of data rows in a CSV file.
    """
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        return sum(1 for _ in reader)


def main():

    print("Creating dataset preparer...")

    preparer = DatasetPreparer(
        CharacterVocabularyBuilder()
    )


    print("Building vocabulary...")

    train_reader = CSVReader(
        TRAIN_FILE
    )

    train_rows = count_csv_rows(TRAIN_FILE)

    vocabulary = preparer.prepare_vocabulary(
        tqdm(
            train_reader.read(),
            total=train_rows,
            desc="Building vocabulary",
        ),
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
        tqdm(
            train_reader.read(),
            total=train_rows,
            desc="Encoding train dataset",
        ),
        tokenizer,
        TRAIN_BIN,
    )


    print("Encoding validation dataset...")

    validation_reader = CSVReader(
        VALIDATION_FILE
    )

    validation_rows = count_csv_rows(VALIDATION_FILE)

    preparer.encode_to_file(
        tqdm(
            validation_reader.read(),
            total=validation_rows,
            desc="Encoding validation dataset",
        ),
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

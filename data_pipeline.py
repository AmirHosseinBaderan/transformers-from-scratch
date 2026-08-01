from pathlib import Path

from torch.utils.data import DataLoader

from common.configs.data_config import DataConfig
from common.data.builders.character_vocabulary_builder import (
    CharacterVocabularyBuilder,
)
from common.data.character_tokenizer import CharacterTokenizer
from common.data.readers.csv_reader import CSVReader
from common.data.dataset import TextDataset

reader = CSVReader(
    Path("common/data/raw/train.csv")
)

texts = reader.read()

builder = CharacterVocabularyBuilder()

vocabulary = builder.build(
    "\n".join(texts)
)
tokenizer = CharacterTokenizer(vocabulary)
tokens = tokenizer.encode(
    "\n".join(texts)
)
dataset = TextDataset(
    token_ids=tokens,
    block_size=DataConfig.BLOCK_SIZE,
)
loader = DataLoader(
    dataset,
    batch_size=DataConfig.TRAIN_BATCH_SIZE,
    shuffle=True,
)
x, y = next(iter(loader))

print(x.shape)
print(y.shape)
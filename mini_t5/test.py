from mini_t5.modules.tokenizer import CharacterTokenizer
from mini_t5.modules.dataset import TranslationDataset


texts = [
    "سلام",
    "hello",
]


tokenizer = CharacterTokenizer()

tokenizer.fit(texts)


dataset = TranslationDataset(
    "./common/data/raw/en_fa_translation_dataset.csv",
    tokenizer,
)


item = dataset[0]


print(item)

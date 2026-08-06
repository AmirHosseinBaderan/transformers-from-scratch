from mini_t5.modules.tokenizer import CharacterTokenizer

texts = [
    "سلام",
    "hello",
    "good morning",
]

tokenizer = CharacterTokenizer()

tokenizer.fit(texts)

ids = tokenizer.encode("سلام")

print(ids)

print(tokenizer.decode(ids))
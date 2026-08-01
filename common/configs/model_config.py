from pathlib import Path


class ModelConfig:

    VOCAB_PATH = Path(
        "common/data/processed/vocab.json"
    )


    EMBEDDING_DIM = 128

    BLOCK_SIZE = 128

    NUM_HEADS = 4

    NUM_LAYERS = 4

    DROPOUT = 0.1
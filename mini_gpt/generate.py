from __future__ import annotations

from pathlib import Path

import torch

from common.configs.model_config import ModelConfig
from common.configs.data_config import DataConfig

from common.data.vocabulary import Vocabulary
from common.data.character_tokenizer import CharacterTokenizer

from common.utils.logger import logger

from mini_gpt.config import GPTConfig
from mini_gpt.model import MiniGPT


def load_model(
    checkpoint_path,
    device,
    vocabulary_size,
):

    model = MiniGPT(
        vocab_size=vocabulary_size,
        block_size=DataConfig.BLOCK_SIZE,
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        num_heads=ModelConfig.NUM_HEADS,
        num_layers=ModelConfig.NUM_LAYERS,
        dropout=ModelConfig.DROPOUT,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    vocabulary = Vocabulary.load(
        ModelConfig.VOCAB_PATH
    )


    tokenizer = CharacterTokenizer(
        vocabulary
    )


    best_model_path = Path(
        GPTConfig.CHECKPOINT_DIR
    ) / "best_model.pt"

    model = load_model(
        checkpoint_path=best_model_path,
        device=device,
        vocabulary_size=len(vocabulary),
    )


    prompt = "Once upon a time"


    input_ids = tokenizer.encode(
        prompt
    )


    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)


    generated = model.generate(
        input_tensor,
        max_new_tokens=100,
        temperature=0.8,
        do_sample=True,
    )


    generated_ids = generated[0].tolist()


    text = tokenizer.decode(
        generated_ids
    )


    logger.info("=" * 50)
    logger.info(text)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
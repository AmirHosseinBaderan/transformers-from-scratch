from __future__ import annotations

from pathlib import Path

import torch

from common.configs.model_config import ModelConfig
from common.configs.data_config import DataConfig

from common.data.vocabulary import Vocabulary
from common.data.character_tokenizer import CharacterTokenizer
from common.data.preprocessing.masker import Masker

from common.utils.logger import logger

from mini_bert.model import MiniBERT
from mini_bert.tasks import MaskedLMHead


def load_model(
    checkpoint_path,
    device,
    vocabulary_size,
):

    model = MiniBERT(
        vocab_size=vocabulary_size,
        block_size=DataConfig.BLOCK_SIZE,
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        num_layers=ModelConfig.NUM_LAYERS,
        dropout=ModelConfig.DROPOUT,
    )

    mlm_head = MaskedLMHead(
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        vocab_size=vocabulary_size,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    mlm_head.load_state_dict(
        checkpoint["mlm_head_state_dict"]
    )

    model.to(device)
    mlm_head.to(device)

    model.eval()
    mlm_head.eval()

    return model, mlm_head


def predict_masked(
    model,
    mlm_head,
    masker,
    tokenizer,
    prompt: str,
    device,
):
    """
    Predict masked tokens in a prompt using the trained BERT model.

    Args:
        model: The BERT model.
        mlm_head: The MaskedLMHead.
        masker: The Masker instance for creating masked inputs.
        tokenizer: The CharacterTokenizer for encoding/decoding.
        prompt: Input text that may contain <MASK> tokens.
        device: Device to run on.

    Returns:
        The decoded text with predicted tokens filled in.
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)

    # If no <MASK> token in the prompt, mask random tokens
    mask_id = tokenizer.vocabulary.token_to_id("<MASK>")
    has_mask = mask_id in input_ids

    if not has_mask:
        # Mask 15% of tokens randomly
        import random
        masked_ids = input_ids.copy()
        labels = [-100] * len(input_ids)
        for i, token_id in enumerate(input_ids):
            if random.random() < 0.15:
                masked_ids[i] = mask_id
                labels[i] = token_id
        input_ids = masked_ids

    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        hidden_states = model(input_tensor)
        logits = mlm_head(hidden_states)

    # Get predictions for masked positions
    mask_positions = [
        i for i, tid in enumerate(input_ids) if tid == mask_id
    ]

    predicted_ids = input_ids.copy()
    for pos in mask_positions:
        token_logits = logits[0, pos]
        predicted_id = token_logits.argmax(dim=-1).item()
        predicted_ids[pos] = predicted_id

    predicted_text = tokenizer.decode(predicted_ids)
    original_text = tokenizer.decode(input_ids)

    return original_text, predicted_text, mask_positions


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
        vocabulary,
    )

    masker = Masker(
        vocabulary=vocabulary,
        mask_probability=0.15,
    )

    best_model_path = Path(
        ModelConfig.CHECKPOINT_DIR
    ) / "best_model.pt"

    model, mlm_head = load_model(
        checkpoint_path=best_model_path,
        device=device,
        vocabulary_size=len(vocabulary),
    )

    prompts = [
        "The cat sat on the <MASK>",
        "Hello <MASK> world",
        "This is a <MASK> test",
    ]

    logger.info("=" * 60)
    logger.info("BERT MLM Prediction")
    logger.info("=" * 60)

    for prompt in prompts:
        original, predicted, positions = predict_masked(
            model=model,
            mlm_head=mlm_head,
            masker=masker,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
        )

        logger.info(f"Prompt:    {prompt}")
        logger.info(f"Predicted: {predicted}")
        logger.info(f"Mask positions: {positions}")
        logger.info("-" * 60)


if __name__ == "__main__":
    main()

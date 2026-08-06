import torch

from mini_t5.config import T5Config
from mini_t5.model import MiniT5
from mini_t5.modules.tokenizer import CharacterTokenizer

from common.utils.logger import logger


def load_best_model():
    """Load the best trained model and tokenizer for inference."""
    # Build tokenizer from training data
    import pandas as pd
    train_df = pd.read_csv(T5Config.TRAIN_CSV_PATH)
    val_df = pd.read_csv(T5Config.VAL_CSV_PATH)

    texts = []
    texts.extend(train_df["source"].astype(str).tolist())
    texts.extend(train_df["target"].astype(str).tolist())
    texts.extend(val_df["source"].astype(str).tolist())
    texts.extend(val_df["target"].astype(str).tolist())

    tokenizer = CharacterTokenizer()
    tokenizer.fit(texts)

    # Build model
    model = MiniT5(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=T5Config.EMBEDDING_DIM,
        num_layers=T5Config.NUM_LAYERS,
        num_heads=T5Config.NUM_HEADS,
        ff_hidden_dim=T5Config.EMBEDDING_DIM * 4,
        max_length=T5Config.MAX_LENGTH,
        dropout=T5Config.DROPOUT,
        use_gradient_checkpointing=False,
    )

    # Load best checkpoint
    best_path = T5Config.CHECKPOINT_DIR + "/best_model.pt"
    try:
        checkpoint = torch.load(best_path, map_location=T5Config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        logger.info(f"Loaded best model from {best_path} (epoch {epoch})")
    except FileNotFoundError:
        logger.error(f"Best model not found at {best_path}. Please train the model first.")
        raise

    model.to(T5Config.DEVICE)
    model.eval()

    return model, tokenizer


def translate_text(model, tokenizer, text):
    """Translate a single text using the model."""
    translator = T5Translator(model, tokenizer, T5Config.DEVICE)
    return translator.translate(text)


class T5Translator:
    """Wrapper for T5 translation inference."""

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def translate(self, text, max_length=64):
        """Translate text from English to Farsi."""
        # Encoder input
        encoder_ids = self.tokenizer.encode(text)
        encoder_ids = torch.tensor(encoder_ids, dtype=torch.long)
        encoder_ids = encoder_ids.unsqueeze(0).to(self.device)

        # Start decoder
        decoder_ids = [self.tokenizer.bos_id]

        for _ in range(max_length):
            decoder_tensor = torch.tensor(decoder_ids, dtype=torch.long)
            decoder_tensor = decoder_tensor.unsqueeze(0).to(self.device)

            logits = self.model(encoder_ids, decoder_tensor)

            next_token_logits = logits[0, -1]
            next_token = torch.argmax(next_token_logits).item()

            decoder_ids.append(next_token)

            if next_token == self.tokenizer.eos_id:
                break

        return self.tokenizer.decode(decoder_ids)


def main():
    logger.info("=" * 60)
    logger.info("MiniT5 English to Farsi Translator")
    logger.info("=" * 60)

    # Load model
    model, tokenizer = load_best_model()

    print("\nEnter English text to translate to Farsi.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("English: ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not user_input:
                continue

            translation = translate_text(model, tokenizer, user_input)
            print(f"Farsi:  {translation}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Translation error: {e}")


if __name__ == "__main__":
    main()

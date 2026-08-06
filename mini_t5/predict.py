import torch

from mini_t5.config import T5Config
from mini_t5.model import MiniT5
from mini_t5.modules.tokenizer import CharacterTokenizer

from common.utils.logger import logger


def load_best_model():
    """
    Load trained MiniT5 model and tokenizer.
    """

    tokenizer = CharacterTokenizer()

    tokenizer.load(
        T5Config.TOKENIZER_PATH
    )

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

    checkpoint_path = (
        T5Config.CHECKPOINT_DIR
        + "/best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=T5Config.DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(
        T5Config.DEVICE
    )

    model.eval()

    epoch = checkpoint.get(
        "epoch",
        "unknown"
    )

    logger.info(
        f"Loaded model epoch={epoch}"
    )

    return model, tokenizer


class T5Translator:

    def __init__(
        self,
        model,
        tokenizer,
        device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(device)
        self.model.eval()


    @torch.no_grad()
    def translate(
        self,
        text,
        max_length=64,
    ):

        # Encoder input
        encoder_ids = self.tokenizer.encode(
            text
        )

        encoder_tensor = torch.tensor(
            encoder_ids,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)


        # Encoder padding mask
        encoder_mask = (
            encoder_tensor != self.tokenizer.pad_id
        ).unsqueeze(1).unsqueeze(2)


        # Decoder starts with BOS
        decoder_ids = [
            self.tokenizer.bos_id
        ]


        for _ in range(max_length):

            decoder_tensor = torch.tensor(
                decoder_ids,
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)


            logits = self.model(
                encoder_tensor,
                decoder_tensor,
                encoder_mask=encoder_mask,
            )


            next_token_logits = logits[
                0,
                -1
            ]


            # Greedy decoding
            next_token = torch.argmax(
                next_token_logits
            ).item()


            decoder_ids.append(
                next_token
            )


            if next_token == self.tokenizer.eos_id:
                break


        return self.decode_output(
            decoder_ids
        )


    def decode_output(
        self,
        ids
    ):

        output = []


        for idx in ids:

            if idx == self.tokenizer.bos_id:
                continue


            if idx == self.tokenizer.eos_id:
                break


            if idx == self.tokenizer.pad_id:
                continue


            output.append(idx)


        return self.tokenizer.decode(
            output
        )


def translate_text(
    model,
    tokenizer,
    text,
):

    translator = T5Translator(
        model,
        tokenizer,
        T5Config.DEVICE,
    )

    return translator.translate(
        text
    )


def main():

    logger.info("=" * 60)
    logger.info("MiniT5 English To Farsi Translator")
    logger.info("=" * 60)


    model, tokenizer = load_best_model()


    translator = T5Translator(
        model,
        tokenizer,
        T5Config.DEVICE
    )


    while True:

        try:

            text = input(
                "\nEnglish: "
            ).strip()


            if text.lower() in [
                "quit",
                "exit",
                "q"
            ]:
                break


            if not text:
                continue


            result = translator.translate(
                text
            )


            print(
                f"Farsi: {result}"
            )


        except KeyboardInterrupt:
            break


        except Exception as e:
            logger.error(
                str(e)
            )


if __name__ == "__main__":
    main()
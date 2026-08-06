import torch
import torch.nn.functional as F

from mini_t5.config import T5Config
from mini_t5.model import MiniT5
from mini_t5.modules.tokenizer import CharacterTokenizer

from common.utils.logger import logger



def load_best_model():
    """
    Load trained MiniT5 model and tokenizer.
    """

    # Load tokenizer from training
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
        device="cpu"
    ):

        self.model = model

        self.tokenizer = tokenizer

        self.device = device


        self.model.to(
            device
        )

        self.model.eval()



    @torch.no_grad()
    def translate(
        self,
        text,
        max_length=64,
        temperature=0.8,
    ):


        encoder_ids = self.tokenizer.encode(
            text
        )


        encoder_ids = torch.tensor(
            encoder_ids,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)



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
                encoder_ids,
                decoder_tensor
            )


            next_logits = logits[
                0,
                -1
            ]



            # جلوگیری از تکرار شدید
            for token in set(decoder_ids):

                next_logits[token] -= 1.5



            probs = F.softmax(
                next_logits / temperature,
                dim=-1
            )


            next_token = torch.multinomial(
                probs,
                num_samples=1
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

        tokens = []


        for idx in ids:

            if idx == self.tokenizer.bos_id:
                continue


            if idx == self.tokenizer.eos_id:
                break


            if idx == self.tokenizer.pad_id:
                continue


            tokens.append(
                idx
            )


        return self.tokenizer.decode(
            tokens
        )





def translate_text(
    model,
    tokenizer,
    text
):

    translator = T5Translator(
        model,
        tokenizer,
        T5Config.DEVICE
    )


    return translator.translate(
        text
    )





def main():

    logger.info(
        "=" * 60
    )

    logger.info(
        "MiniT5 English To Farsi Translator"
    )

    logger.info(
        "=" * 60
    )



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
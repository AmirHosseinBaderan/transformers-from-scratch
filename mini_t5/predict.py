import torch



class T5Translator:


    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
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


        encoder_ids = torch.tensor(
            encoder_ids,
            dtype=torch.long
        )


        encoder_ids = encoder_ids.unsqueeze(0)


        encoder_ids = encoder_ids.to(
            self.device
        )



        # Start decoder

        decoder_ids = [
            self.tokenizer.bos_id
        ]



        for _ in range(max_length):


            decoder_tensor = torch.tensor(
                decoder_ids,
                dtype=torch.long
            )


            decoder_tensor = (
                decoder_tensor
                .unsqueeze(0)
                .to(self.device)
            )



            logits = self.model(
                encoder_ids,
                decoder_tensor
            )



            next_token_logits = logits[
                0,
                -1
            ]



            next_token = torch.argmax(
                next_token_logits
            ).item()



            decoder_ids.append(
                next_token
            )



            if next_token == self.tokenizer.eos_id:

                break



        return self.tokenizer.decode(
            decoder_ids
        )
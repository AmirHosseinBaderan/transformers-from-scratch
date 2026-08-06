import json


class CharacterTokenizer:

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    @property
    def pad_id(self):
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def bos_id(self):
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self):
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def unk_id(self):
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def fit(self, texts):

        vocabulary = set()

        for text in texts:
            vocabulary.update(text)

        tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        ]

        tokens.extend(sorted(vocabulary))

        self.token_to_id = {
            token: idx
            for idx, token in enumerate(tokens)
        }

        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }

    def encode(
        self,
        text,
        add_special_tokens=True,
    ):

        ids = []

        if add_special_tokens:
            ids.append(self.bos_id)

        for ch in text:
            ids.append(
                self.token_to_id.get(
                    ch,
                    self.unk_id,
                )
            )

        if add_special_tokens:
            ids.append(self.eos_id)

        return ids

    def decode(
        self,
        ids,
        skip_special_tokens=True,
    ):

        tokens = []

        special_tokens = {
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        }

        for idx in ids:

            token = self.id_to_token.get(
                idx,
                self.UNK_TOKEN,
            )

            if (
                skip_special_tokens
                and token in special_tokens
            ):
                continue

            tokens.append(token)

        return "".join(tokens)

    def save(
        self,
        path,
    ):

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.token_to_id,
                f,
                ensure_ascii=False,
                indent=4,
            )

    def load(
        self,
        path,
    ):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            self.token_to_id = json.load(f)

        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }
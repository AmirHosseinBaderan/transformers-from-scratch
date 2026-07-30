from __future__ import annotations

import json
from pathlib import Path


class Vocabulary:
    """
        A generic vocabulary that maps tokens to integer IDs and vice versa.

        The vocabulary is tokenizer-agnostic, meaning it can be used with
        character-level, word-level, BPE, SentencePiece, or any other tokenizer.
    """

    def __init__(self, special_tokens: list[str] | None = None):
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._special_tokens: list[str] = []

        if special_tokens is not None:
            self.add_tokens(special_tokens)
            self._special_tokens = list(special_tokens)

    def __len__(self):
        return len(self._token_to_id)

    def __contains__(self, item):
        return item in self._token_to_id

    def add_tokens(self, token):
        """
        Add a token to the vocabulary.

        Returns the token ID. If the token already exists,
        its existing ID is returned.
        """

        if not isinstance(token, str):
            raise TypeError("token must be a string")

        if token == "":
            raise ValueError("token cannot be an empty string")

        if token in self._token_to_id:
            return self._token_to_id[token]

        token_id = len(self)
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token

        return token_id

    def token_to_id(
            self,
            token: str,
    ) -> int:
        """
        Convert a token to its ID.

        If the token does not exist and an <UNK> token is available,
        the ID of <UNK> is returned.
        """

        if token in self._token_to_id:
            return self._token_to_id[token]

        if "<UNK>" in self._token_to_id:
            return self._token_to_id["<UNK>"]

        raise KeyError(f"Unknown token: {token}")

    def id_to_token(self, token_id):
        try:
            return self._id_to_token[token_id]
        except KeyError as exc:
            raise KeyError(f"Unknown token id: {token_id}") from exc

    def encode(self, tokens):
        return [self.token_to_id(token) for token in tokens]

    def decode(self, token_ids):
        return [self.id_to_token(token_id) for token_id in token_ids]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "special_tokens": self._special_tokens,
            "token_to_id": self._token_to_id,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls,path:Path) -> "Vocabulary":
        with path.open("r",encoding="utf-8") as f:
            data = json.load(f)

        vocabulary = cls(
            special_tokens=data.get("special_tokens", []),
        )
        vocabulary._token_to_id.clear()
        vocabulary._id_to_token.clear()

        for token,token_id in data["token_to_id"].items():
            vocabulary._token_to_id[token] = token_id
            vocabulary._id_to_token[token_id] = token

        return vocabulary
import torch
from torch import nn


import math

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.scale = math.sqrt(embedding_dim)

    def forward(self, input_ids):
        return self.embedding(input_ids) * self.scale
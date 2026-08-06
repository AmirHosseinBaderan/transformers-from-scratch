import torch
from torch import nn



class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        dropout=0.1,
    ):
        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),


            nn.GELU(),


            nn.Dropout(
                dropout
            ),


            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),


            nn.Dropout(
                dropout
            )

        )



    def forward(
        self,
        x,
    ):

        return self.network(x)
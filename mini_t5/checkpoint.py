import torch
import os



class CheckpointManager:


    def __init__(
        self,
        directory="checkpoints"
    ):

        self.directory = directory


        os.makedirs(
            directory,
            exist_ok=True
        )



    def save(
        self,
        model,
        optimizer,
        epoch,
        loss,
        filename="last.pt"
    ):


        path = os.path.join(
            self.directory,
            filename
        )


        checkpoint = {

            "epoch": epoch,

            "loss": loss,

            "model_state_dict":
                model.state_dict(),


            "optimizer_state_dict":
                optimizer.state_dict()

        }


        torch.save(
            checkpoint,
            path
        )


        print(
            f"Checkpoint saved: {path}"
        )



    def load(
        self,
        model,
        optimizer,
        filename="last.pt"
    ):


        path = os.path.join(
            self.directory,
            filename
        )


        checkpoint = torch.load(
            path,
            map_location="cpu"
        )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )


        epoch = checkpoint[
            "epoch"
        ]


        loss = checkpoint[
            "loss"
        ]


        print(
            f"Loaded checkpoint from epoch {epoch}"
        )


        return epoch, loss
import pandas as pd
import torch

from torch.utils.data import Dataset


class TranslationDataset(Dataset):

    def __init__(
        self,
        csv_path,
        tokenizer,
        max_length=64,
    ):
        self.data = pd.read_csv(
            csv_path
        )

        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):

        return len(self.data)


    def pad_or_truncate(
        self,
        ids,
    ):

        if len(ids) > self.max_length:

            ids = ids[:self.max_length]


        padding_length = (
            self.max_length - len(ids)
        )


        ids += [
            self.tokenizer.pad_id
        ] * padding_length


        return ids



    def create_decoder_inputs(
        self,
        target_ids,
    ):

        decoder_input_ids = target_ids[:-1]

        labels = target_ids[1:]


        return (
            decoder_input_ids,
            labels
        )



    def __getitem__(
        self,
        index,
    ):

        row = self.data.iloc[index]


        source = str(
            row["source"]
        )

        target = str(
            row["target"]
        )


        encoder_ids = self.tokenizer.encode(
            source
        )


        target_ids = self.tokenizer.encode(
            target
        )


        decoder_ids, labels = self.create_decoder_inputs(
            target_ids
        )


        encoder_ids = self.pad_or_truncate(
            encoder_ids
        )

        decoder_ids = self.pad_or_truncate(
            decoder_ids
        )

        labels = self.pad_or_truncate(
            labels
        )


        return {

            "encoder_input_ids":
                torch.tensor(
                    encoder_ids,
                    dtype=torch.long
                ),


            "decoder_input_ids":
                torch.tensor(
                    decoder_ids,
                    dtype=torch.long
                ),


            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.long
                )

        }
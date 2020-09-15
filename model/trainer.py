import pathlib
from typing import Any, List, Optional, Tuple

import bpemb
import h5py
import numpy
import PIL
import torch
import torchvision

from utils import create_pad_collate, log_init


class Trainer:

    @log_init
    def __init__(self):
        self.dataset = IGDataset()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=create_pad_collate(pad_token_id=0)
        )
        self.model = Baseline()

    def run(self) -> None:
        for x in self.dataloader:
            y = self.model(x)
            print(y)


class Baseline(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = ResnetEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        images, captions = x
        encoder_output = self.encoder(images)
        caption = self.decoder(encoder_output, captions)

        return caption


class ResnetEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # TODO: export to config
        self.input_height = 320
        self.input_width = 320
        self.hidden_dim = 256

        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.encoder = torch.nn.Sequential(*modules)
        self.linear = torch.nn.Linear(
            in_features=2048*10*10,
            out_features=self.hidden_dim
        )
        self.relu = torch.nn.ReLU()

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)

        return x


class LSTMDecoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # TODO: config
        self.tokenizer = bpemb.BPEmb(
            lang="en",
            vs=100_000,
            dim=300
        )
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(self.tokenizer.vectors)
        )

        self.lstm = torch.nn.LSTM(
            input_size=300,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        captions: torch.Tensor

    ) -> torch.Tensor:
        captions = self.embedding(captions)

        x = self.lstm(
            captions,
            (
                torch.unsqueeze(encoder_output, 0),
                torch.unsqueeze(encoder_output, 0)
            )
        )
        print(x[0].shape)
        print(x[1][0].shape)
        print(x[1][1].shape)
        exit()

        return x


class IGDataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()

        # TODO: config
        self.hdf5_path = pathlib.Path("./.cache/data/instagram/ig_sample.hdf5")
        self.image_directory_path = pathlib.Path("./.cache/data/instagram/ig_sample/images")
        self.split = "train"
        self.image_transformations = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=320),
                torchvision.transforms.CenterCrop(size=320),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        with h5py.File(self.hdf5_path, "r") as hdf5_store:
            hdf5_group = hdf5_store.get(self.split)

            self.caption_ids = numpy.array(
                hdf5_group["caption_id"]
            )

            self.token_ids = numpy.array(
                hdf5_group["caption_cleaned_tokenized_id"]
            )

        assert len(self.caption_ids) == len(self.token_ids)

        self.length = len(self.caption_ids)

    def __getitem__(
        self,
        index: int
    ) -> Tuple[PIL.Image.Image, Optional[numpy.ndarray]]:
        image = PIL.Image.open(
            self.image_directory_path / self.caption_ids[index]
        )
        image = self.image_transformations(image)

        caption_tokenized_id = torch.LongTensor(self.token_ids[index])

        return (image, caption_tokenized_id)

    def __len__(self) -> int:
        return self.length

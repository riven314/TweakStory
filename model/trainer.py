import pathlib
from typing import List, Optional, Tuple

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
        self.hidden_dim = 300

        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.encoder = torch.nn.Sequential(*modules)

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images)
        x = x.view(x.size(0), -1, x.size(1))

        return x


class LSTM(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        context_vector_size: int,
        num_layers: int
    ):
        """
        TODO
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_vector_size = context_vector_size
        self.combined_vector_size = (
            input_size +
            hidden_size +
            context_vector_size
        )
        self.num_layers = num_layers

        if(self.num_layers != 1):
            raise NotImplementedError(
                "Currently, only 1 layer LSTMs are supported."
            )

        # set up gates
        self.input_gate = torch.nn.Linear(
            in_features=self.combined_vector_size,
            out_features=self.hidden_size,
            bias=True
        )
        self.forget_gate = torch.nn.Linear(
            in_features=self.combined_vector_size,
            out_features=self.hidden_size,
            bias=True
        )
        self.output_gate = torch.nn.Linear(
            in_features=self.combined_vector_size,
            out_features=self.hidden_size,
            bias=True
        )
        self.input_layer = torch.nn.Linear(
            in_features=self.combined_vector_size,
            out_features=self.hidden_size,
            bias=True
        )

        # set up projections
        self.context_projection = torch.nn.Linear(
            in_features=self.context_vector_size,
            out_features=self.input_size
        )
        self.hidden_state_projection = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.input_size
        )
        self.output_projection = torch.nn.Linear(
            in_features=self.input_size,
            out_features=self.input_size
        )

    def forward(
        self,
        previous_embedded_caption: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        previous_cell_state: torch.Tensor,
        current_context_vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO
        """
        combined_vector = torch.cat(
            [
                previous_embedded_caption,
                previous_hidden_state,
                current_context_vector
            ],
            dim=1
        )

        # shape: (batch_size, hidden_dimension)
        input_gate_vector = torch.nn.functional.sigmoid(
            self.input_gate(
                combined_vector
            )
        )

        # shape: (batch_size, hidden_dimension)
        forget_gate_vector = torch.nn.functional.sigmoid(
            self.forget_gate(
                combined_vector
            )
        )

        # shape: (batch_size, hidden_dimension)
        output_gate_vector = torch.nn.functional.sigmoid(
            self.output_gate(
                combined_vector
            )
        )

        # shape: (batch_size, hidden_dimension)
        input_vector = torch.nn.functional.tanh(
            self.input_layer(
                combined_vector
            )
        )

        # c_{t} = (
        #           f_{t} <Hadamard product> c_{t-1} +
        #           i_{t} <Hadamard product> g_{t}
        # )
        current_cell_state = torch.add(
            torch.einsum(
                "bh,bh->bh",
                forget_gate_vector,
                previous_cell_state
            ),
            torch.einsum(
                "bh,bh->bh",
                input_gate_vector,
                input_vector
            )
        )

        # h_{t} = o_{t} <Hadamard product> tanh(c_{t})
        current_hidden_state = torch.einsum(
            "bh,bh->bh",
            output_gate_vector,
            torch.nn.functional.tanh(current_cell_state)
        )

        prediction_vector = self.output_projection(
            torch.add(
                torch.add(
                    previous_embedded_caption,
                    self.hidden_state_projection(current_hidden_state)
                ),
                self.context_projection(current_context_vector)
            )
        )

        return (
            prediction_vector,
            current_hidden_state,
            current_cell_state
        )


class LSTMDecoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # TODO: config
        self.hidden_dimension = 300
        self.embedding_dimension = 300
        self.vocabulary_size = 100_000
        self.encoder_output_dimension = 2048
        self.number_of_lstm_layers = 1
        # <unk> token id (unkown)
        self.unk_token_id = 0
        # <s> token id (beginning of sentence)
        self.bos_token_id = 1
        # </s> token id (end of sentence)
        self.eos_token_id = 2

        self.tokenizer = bpemb.BPEmb(
            lang="en",
            vs=self.vocabulary_size,
            dim=self.embedding_dimension
        )
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(self.tokenizer.vectors)
        )
        self.lstm = LSTM(
            input_size=self.embedding_dimension,
            hidden_size=self.hidden_dimension,
            num_layers=self.number_of_lstm_layers,
            context_vector_size=self.encoder_output_dimension
        )
        self.hidden_initializer = torch.nn.Linear(
            in_features=self.encoder_output_dimension,
            out_features=self.hidden_dimension
        )
        self.cell_initializer = torch.nn.Linear(
            in_features=self.encoder_output_dimension,
            out_features=self.hidden_dimension
        )

        self.energy_function = torch.nn.Linear(
            in_features=(
                self.encoder_output_dimension +
                self.hidden_dimension
            ),
            out_features=1
        )

    def get_attention_vector(
        self,
        encoder_output: torch.Tensor,
        previous_hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """
        TODO
        """
        batch_size = encoder_output.size(0)
        channel_length = encoder_output.size(1)

        # TODO: check Einstein summation
        energy = torch.zeros(
            [
                batch_size,
                channel_length
            ],
            dtype=torch.float32
        )
        for channel in range(channel_length):
            energy[:, channel] = torch.squeeze(
                self.energy_function(
                    torch.cat(
                        (
                            encoder_output[:, channel, :],
                            previous_hidden_state
                        ),
                        dim=1
                    )
                )
            )
        # TODO
        return torch.nn.functional.softmax(energy, dim=1)

    def get_context_vector(
        self,
        encoder_output: torch.Tensor,
        attention_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        TODO
        """
        context_vector = torch.einsum(
            "bcf,bcf->bf",
            attention_vector.unsqueeze(2).expand_as(encoder_output),
            encoder_output
        )

        return context_vector

    def forward(
        self,
        encoder_output: torch.Tensor,
        captions: torch.Tensor
    ) -> torch.Tensor:
        """
        param encoder_output: Encounter output.
          Shape: (batch_size, channels, feature_dimension)
        param captions: Captions encoded as padded tensors of token ids.
          Shape: (batch_size, padded_length)

        return: TODO
        """
        # TODO
        teacher_forcing = True
        batch_size = captions.size(0)
        padded_length = captions.size(1)
        channel_length = encoder_output.size(1)

        # shape: (batch_size, padded_length, word_embedding_dimension)
        embedded_captions_tensor = self.embedding(captions)

        prediction_tensor = torch.zeros(
            [
                batch_size,
                padded_length,
                self.embedding_dimension
            ],
            dtype=torch.float32
        )
        # set first entry of prediction to <s>
        prediction_tensor[:, 0, :] = torch.from_numpy(
            self.tokenizer.vectors[self.bos_token_id]
        )
        hidden_state_tensor = torch.zeros(
            [
                batch_size,
                padded_length,
                self.hidden_dimension
            ],
            dtype=torch.float32
        )
        cell_state_tensor = torch.zeros(
            [
                batch_size,
                padded_length,
                self.hidden_dimension
            ],
            dtype=torch.float32
        )
        attention_weight_tensor = torch.zeros(
            [
                batch_size,
                padded_length,
                channel_length
            ],
            dtype=torch.float32
        )
        context_tensor = torch.zeros(
            [
                batch_size,
                padded_length,
                self.encoder_output_dimension
            ],
            dtype=torch.float32
        )

        (
            hidden_state_tensor[:, 0, :],
            cell_state_tensor[:, 0, :]
        ) = self.initialize_states(encoder_output)

        for time_step in range(1, padded_length):
            # create current context vector
            attention_weight_tensor[:, time_step, :] = (
                self.get_attention_vector(
                    encoder_output,
                    hidden_state_tensor[:, time_step-1, :]
                )
            )
            context_tensor[:, time_step, :] = self.get_context_vector(
                encoder_output,
                attention_weight_tensor[:, time_step, :]
            )

            # perform one step in LSTM
            if teacher_forcing:
                previous_embedded_caption = (
                    embedded_captions_tensor[:, time_step-1, :]
                )
            else:
                previous_embedded_caption = (
                    prediction_tensor[:, time_step-1, :]
                )

            (
                prediction_tensor[:, time_step, :],
                hidden_state_tensor[:, time_step, :],
                cell_state_tensor[:, time_step, :]
            ) = self.lstm(
                previous_embedded_caption,
                hidden_state_tensor[:, time_step-1, :],
                cell_state_tensor[:, time_step-1, :],
                context_tensor[:, time_step, :]
            )

        return prediction_tensor

    def initialize_states(
        self,
        encoder_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell state for a single layer LSTM.

        param encoder_output: Encounter output.
          Shape: (batch_size, channels, feature_dimension)

        return: (hidden_state_0, cell_state_0)
         Shape ((batch_size, hidden_dimension), (batch_size, hidden_dimension))
        """
        mean_encoder_output = torch.mean(encoder_output, dim=1)
        hidden_state_0 = torch.nn.functional.tanh(
            self.hidden_initializer(mean_encoder_output)
        )
        cell_state_0 = torch.nn.functional.tanh(
            self.cell_initializer(mean_encoder_output)
        )

        return (hidden_state_0, cell_state_0)


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

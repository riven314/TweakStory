import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

import bpemb  # type: ignore
import h5py  # type: ignore
import numpy  # type: ignore
import torch
import torchvision  # type: ignore
import tqdm  # type: ignore

from utils import create_pad_collate, get_image_transformations, log_init


class Trainer:
    """
    Trainer trains an 'image -> caption' model.
    """

    @log_init
    def __init__(
        self,
        *,
        config: Dict[str, Any]
    ):
        """
        Initialize Trainer.

        :param config: Configuration of Trainer.
          Example:
          {
              "pad_token_id": 0,
              "unk_token_id": 0,
              "batch_size": 16,
              "max_epochs": 100,
              "device_name": "cpu",
              "dataloader_num_workers": 2,
              "IGDataset": {
                "cache_location": "./.cache/data/instagram/",
                "dataset_name": "ig_sample",
                "split": "train",
                "image_size": 320,
                "crop_size": 320,
                "image_normalization_mean": [
                  0.485,
                  0.456,
                  0.406
                ],
                "image_normalization_std": [
                  0.229,
                  0.224,
                  0.225
                ]
              },
              "ShowAttendTell": {
                "ResnetEncoder": {
                  "input_height": 320,
                  "input_width": 320,
                  "hidden_dim": 300
                },
                "AttentionLSTMDecoder": {
                  "hidden_dimension": 300,
                  "embedding_dimension": 300,
                  "vocabulary_size": 100000,
                  "encoder_output_dimension": 2048,
                  "number_of_lstm_layers": 1,
                  "unk_token_id": 0,
                  "bos_token_id": 1,
                  "eos_token_id": 2,
                  "language": "en"
                }
              }
          }
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.device = torch.device(self.config["device_name"])

        # training data
        self.dataset = IGDataset(
            config=self.config.get("IGDataset", {})
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["dataloader_num_workers"],
            collate_fn=create_pad_collate(
                pad_token_id=self.config["pad_token_id"]
            ),
            pin_memory=True
        )

        # model
        self.model = ShowAttendTell(
            config=self.config["ShowAttendTell"]
        )
        self.model.to(
            self.device
        )
        self.criterion = torch.nn.NLLLoss(
            ignore_index=self.config["unk_token_id"],
            reduction="mean"
        ).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def run(self) -> None:
        """
        Run Trainer.

        return: None
        """
        self.logger.info("Training model.")

        epoch_loss = 0.
        epoch_length = 0

        for epoch in range(1, self.config["max_epochs"] + 1):
            epoch_loss = 0.
            epoch_length = 0
            self.logger.info(f"Starting epoch: {epoch}")

            for x in tqdm.tqdm(self.dataloader):
                self.optimizer.zero_grad()
                image_representations, captions = x
                image_representations = image_representations.to(self.device)
                captions = captions.to(self.device)
                prediction = self.model.decoder(
                    image_representations,
                    captions
                )
                loss = 0
                padded_length = captions.size(1)

                for index in range(1, padded_length):
                    loss += self.criterion(
                        prediction[:, index, :],
                        captions[:, index]
                    )

                loss.backward()                                 # type: ignore
                self.optimizer.step()
                epoch_loss += loss.item()                       # type: ignore
                epoch_length += 1

            epoch_loss /= epoch_length
            self.logger.info(
                f"Epoch: {epoch:>4} Training Loss: {epoch_loss:4.2f}"
            )


class ShowAttendTell(torch.nn.Module):
    """
    ShowAttendTell is a 'image -> caption' model.

    See: https://arxiv.org/abs/1502.03044
    """

    def __init__(
        self,
        *,
        config: Dict[str, Any]
    ):
        """
        Initialize AttendShowTell.

        :param config: Configuration of ShowAttendTell.
          Example:
          {
            "ResnetEncoder": {
              "input_height": 320,
              "input_width": 320,
              "hidden_dim": 300
            },
            "AttentionLSTMDecoder": {
              "hidden_dimension": 300,
              "embedding_dimension": 300,
              "vocabulary_size": 100000,
              "encoder_output_dimension": 2048,
              "number_of_lstm_layers": 1,
              "unk_token_id": 0,
              "bos_token_id": 1,
              "eos_token_id": 2,
              "language": "en"
            }
          }
        """
        super().__init__()
        self.config = config

        self.encoder = ResnetEncoder(
            config=self.config["ResnetEncoder"]
        )
        self.decoder = AttentionLSTMDecoder(
            config=self.config["AttentionLSTMDecoder"]
        )

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run ShowAttendTell on a single batch.

        param images: Images to be encoded.
          Shape: (batch_size, color_channels, height, width)
        param captions: Target captions.
          Shape: (batch_size, padded_length)

        return: Tensor containing predicted word vectors.
          Shape (batch_size, padded_length, word_embedding_dimension)
        """
        encoder_output = self.encoder(images)
        prediction_tensor = self.decoder(encoder_output, captions)

        return prediction_tensor

    def to(self, *args: Any, **kwargs: Any) -> Any:
        """
        Move this model to the desired device.

        return: Moved version of this model.
        """
        self = super().to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)

        return self


class ResnetEncoder(torch.nn.Module):
    """
    ResnetEncoder encodes a given image via a pretrained ResNet101 model.

    :param config: Configuration of ResnetEncoder.
      Example:
       {
         "input_height": 320,
         "input_width": 320,
         "hidden_dim": 300
       }
    """

    def __init__(
        self,
        *,
        config: Dict[str, Any]
    ):
        """
        Initialize ResnetEncoder.
        """
        super().__init__()
        self.config = config

        self.input_height = self.config["input_height"]
        self.input_width = self.config["input_width"]
        self.hidden_dim = self.config["hidden_dim"]

        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.encoder = torch.nn.Sequential(*modules)

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """
        Initialize parameters.

        Define trainable weights and their initial values.

        return: None
        """
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run ResnetEncoder on a single batch.

        :param image: Tensor of input images.
          Shape: (batch_size, input_height, input_width)

        return: Encoded representation of the given input images.
          Shape: (batch_size, 100, 2048)
        """
        encoder_output = self.encoder(images)
        encoder_output = encoder_output.view(
            encoder_output.size(0),
            -1,
            encoder_output.size(1)
        )

        return encoder_output

    def to(self, *args: Any, **kwargs: Any) -> Any:
        """
        Move this model to the desired device.

        return: Moved version of this model.
        """
        self = super().to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)

        return self


class LSTM(torch.nn.Module):
    """
    Custom LSTM cell that takes a context vector as additional input.

    See: https://arxiv.org/abs/1502.03044
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        context_vector_size: int,
        vocabulary_size: int,
        num_layers: int
    ):
        """
        Initialize LSTM.

        :param input_size: Dimension of word embeddings.
        :param hidden_size: Dimension of hidden state.
        :param context_vector_size: Dimension of context vector.
        :param vocabulary_size: Size of vocabulary.
        :num_layers: Number of stacked LSTM layers.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_vector_size = context_vector_size
        self.vocabulary_size = vocabulary_size
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
            out_features=self.vocabulary_size
        )

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """
        Initialize parameters.

        Define trainable weights and their initial values.

        return: None
        """
        bias_regex = re.compile(r"^.*\.bias$")
        weight_regex = re.compile(r"^.*\.weight")
        for name, parameter in self.named_parameters():
            if re.match(bias_regex, name):
                torch.nn.init.constant_(parameter, .0)
            elif re.match(weight_regex, name):
                torch.nn.init.xavier_normal_(parameter)
            parameter.requires_grad = True

    def forward(
        self,
        previous_embedded_caption: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        previous_cell_state: torch.Tensor,
        current_context_vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run LSTM on a single batch.

        :param previous_embedded_caption: Last known word vector.
          Shape:  (batch_size, word_embedding_dimension)
        :param previous_hidden_state: Last known hidden state.
          Shape: (batch_size, hidden_dimension)
        :param previous_cell_state: Last known cell state.
          Shape: (batch_size, hidden_dimension)
        :param current_context_vector: Current context vector.
          Shape: (batch_size, context_vector_size)

        return: (
              prediction_vector,
              current_hidden_state,
              current_cell_state
          )
          prediction_vector is the log-likelihood vector of the predicted word.
            Shape: (batch_size, vocabulary_size)
          current_hidden_state is the current hidden state.
            Shape: (batch_size, hidden_dimension)
          current_cell_state is the current cell state.
            Shape: (batch_size, hidden_dimension)
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
        input_gate_vector = torch.sigmoid(
            self.input_gate(
                combined_vector
            )
        )

        # shape: (batch_size, hidden_dimension)
        forget_gate_vector = torch.sigmoid(
            self.forget_gate(
                combined_vector
            )
        )

        # shape: (batch_size, hidden_dimension)
        output_gate_vector = torch.sigmoid(
            self.output_gate(
                combined_vector
            )
        )

        # shape: (batch_size, hidden_dimension)
        input_vector = torch.tanh(
            self.input_layer(
                combined_vector
            )
        )

        current_cell_state = torch.add(
            torch.einsum(
                "bh,bh->bh",
                forget_gate_vector,
                previous_cell_state.clone()
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
            torch.tanh(current_cell_state)
        )

        prediction_vector = torch.nn.functional.log_softmax(
            self.output_projection(
                torch.add(
                    torch.add(
                        previous_embedded_caption,
                        self.hidden_state_projection(current_hidden_state)
                    ),
                    self.context_projection(current_context_vector.clone())
                )
            ),
            dim=1
        )

        return (
            prediction_vector,
            current_hidden_state,
            current_cell_state
        )

    def to(self, *args: Any, **kwargs: Any) -> Any:
        """
        Move this model to the desired device.

        return: Moved version of this model.
        """
        self = super().to(*args, **kwargs)
        self.input_gate = self.input_gate.to(*args, **kwargs)
        self.forget_gate = self.forget_gate.to(*args, **kwargs)
        self.output_gate = self.output_gate.to(*args, **kwargs)
        self.input_layer = self.input_layer.to(*args, **kwargs)
        self.context_projection = self.context_projection.to(*args, **kwargs)
        self.hidden_state_projection = self.hidden_state_projection.to(
            *args,
            **kwargs
        )
        self.output_projection = self.output_projection.to(*args, **kwargs)

        return self


class AttentionLSTMDecoder(torch.nn.Module):
    """
    AttentionLSTMDecoder is an LSTM Vec2Seq decoder with additive attention.

    See: https://arxiv.org/abs/1502.03044
    """

    def __init__(
        self,
        *,
        config: Dict[str, Any]
    ):
        """
        Initialize AttentionLSTMDecoder.

        :param config: Configuration of AttentionLSTMDecoder.
          Example:
          {
            "hidden_dimension": 300,
            "embedding_dimension": 300,
            "vocabulary_size": 100000,
            "encoder_output_dimension": 2048,
            "number_of_lstm_layers": 1,
            "unk_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "language": "en"
          }
        """
        super().__init__()
        self.config = config

        self.tokenizer = bpemb.BPEmb(
            lang=self.config["language"],
            vs=self.config["vocabulary_size"],
            dim=self.config["embedding_dimension"]
        )
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(self.tokenizer.vectors)
        )
        self.lstm = LSTM(
            input_size=self.config["embedding_dimension"],
            hidden_size=self.config["hidden_dimension"],
            num_layers=self.config["number_of_lstm_layers"],
            vocabulary_size=self.config["vocabulary_size"],
            context_vector_size=self.config["encoder_output_dimension"]
        )
        self.hidden_initializer = torch.nn.Linear(
            in_features=self.config["encoder_output_dimension"],
            out_features=self.config["hidden_dimension"]
        )
        self.cell_initializer = torch.nn.Linear(
            in_features=self.config["encoder_output_dimension"],
            out_features=self.config["hidden_dimension"]
        )
        self.energy_function = torch.nn.Linear(
            in_features=(
                self.config["encoder_output_dimension"] +
                self.config["hidden_dimension"]
            ),
            out_features=1
        )

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """
        Initialize parameters.

        Define trainable weights and their initial values.

        return: None
        """
        pretrained_regex = re.compile(r"^embedding.*$")
        skip_regex = re.compile(r"^(lstm|embedding).*$")
        bias_regex = re.compile(r"^.*\.bias$")
        weight_regex = re.compile(r"^.*\.weight")
        for name, parameter in self.named_parameters():
            if re.match(pretrained_regex, name):
                parameter.requires_grad = False
            elif re.match(skip_regex, name):
                pass
            elif re.match(bias_regex, name):
                torch.nn.init.constant_(parameter, .0)
                parameter.requires_grad = True
            elif re.match(weight_regex, name):
                torch.nn.init.xavier_normal_(parameter)
                parameter.requires_grad = True

    def get_attention_vector(
        self,
        encoder_output: torch.Tensor,
        previous_hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attention weights.

        :parameter encoder_output: Encoder output.
          Shape: (batch_size, channel_length, embedding_dimension)
        :param previous_hidden_state: Last known hidden state.
          Shape: (batch_size, hidden_dimension)

        return: Attention weights.
          Shape: (batch_size, channel_length)
        """
        channel_length = encoder_output.size(1)

        energies: List[torch.Tensor] = []

        # TODO: Check whether this can be optimzed via vectorization.
        for channel in range(channel_length):
            energies.append(
                torch.squeeze(
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
            )

        energy_vector = torch.stack(energies, dim=1)

        return torch.nn.functional.softmax(energy_vector, dim=1)

    def get_context_vector(
        self,
        encoder_output: torch.Tensor,
        attention_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate context vector.

        :parameter encoder_output: Encoder output.
          Shape: (batch_size, channel_length, embedding_dimension)
        :param attention_vector: Attention weights.
          Shape: (batch_size, channel_length)

        return: Context vector.
          Shape: (batch_size, embedding_dimension)
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
        Run AttentionLSTMDecoder on a single batch.

        param encoder_output: Encounter output.
          Shape: (batch_size, channels, feature_dimension)
        param captions: Captions encoded as padded tensors of token ids.
          Shape: (batch_size, padded_length)

        return: Tensor containing log-likelihood vectors for predicted words.
          Shape: (batch_size, padded_length, vocabulary_size)
        """
        # TODO: Properly implement teacher forcing
        device = encoder_output.device
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
                self.config["vocabulary_size"]
            ],
            dtype=torch.float32
        ).to(
            device
        )
        # set first entry of prediction to <s>
        prediction_tensor[:, 0, self.config["bos_token_id"]] = 1.
        hidden_state_tensor = torch.zeros(
            [
                batch_size,
                padded_length,
                self.config["hidden_dimension"]
            ],
            dtype=torch.float32
        ).to(
            device
        )
        cell_state_tensor = torch.zeros(
            [
                batch_size,
                padded_length,
                self.config["hidden_dimension"]
            ],
            dtype=torch.float32
        ).to(
            device
        )
        attention_weight_tensor = torch.zeros(
            [
                batch_size,
                padded_length,
                channel_length
            ],
            dtype=torch.float32
        ).to(
            device
        )
        context_tensor = torch.zeros(
            [
                batch_size,
                padded_length,
                self.config["encoder_output_dimension"]
            ],
            dtype=torch.float32
        ).to(
            device
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
                # TODO: implement non teacher forcing
                raise NotImplementedError(
                    "Disabling teacher forcing isn't implemented, yet"
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
        hidden_state_0 = torch.tanh(
            self.hidden_initializer(mean_encoder_output)
        )
        cell_state_0 = torch.tanh(
            self.cell_initializer(mean_encoder_output)
        )

        return (hidden_state_0, cell_state_0)

    def to(self, *args: Any, **kwargs: Any) -> Any:
        """
        Move this model to the desired device.

        return: Moved version of this model.
        """
        self = super().to(*args, **kwargs)
        self.embedding = self.embedding.to(*args, **kwargs)
        self.lstm = self.lstm.to(*args, **kwargs)
        self.hidden_initializer = self.hidden_initializer.to(*args, **kwargs)
        self.cell_initializer = self.cell_initializer.to(*args, **kwargs)
        self.energy_function = self.energy_function.to(*args, **kwargs)

        return self


class IGDataset(torch.utils.data.Dataset):
    """
    Instagram Dataset.

    See: IntaPIC-1.1M @ https://github.com/cesc-park/attend2u
    """

    def __init__(
        self,
        *,
        config: Dict[str, Any]
    ):
        """
        Initialize IGDataset.
        """
        super().__init__()

        # TODO: config
        self.config = config
        self.hdf5_path = pathlib.Path(
            self.config["cache_location"] +
            self.config["dataset_name"] +
            ".hdf5"
        )
        self.image_directory_path = pathlib.Path(
            self.config["cache_location"] +
            self.config["dataset_name"] +
            "/images/"
        )
        self.image_transformations = get_image_transformations(
            self.config["image_transformations"]
        )
        with h5py.File(self.hdf5_path, "r") as hdf5_store:
            hdf5_group = hdf5_store.get(self.config["split"])

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
    ) -> Tuple[torch.Tensor, Optional[torch.LongTensor]]:
        """
        Get the i-th item in this dataset.

        :param index: Index of the item to get.

        return: (<image representation>, <tensor of token ids for caption>)
        """
        with h5py.File(self.hdf5_path, "r") as hdf5_store:
            hdf5_group = hdf5_store.get(self.config["split"])
            image_encoded = hdf5_group["image_encoded"]
            image_representation = torch.Tensor(image_encoded[index, :, :])

        caption_tokenized_id = torch.LongTensor(self.token_ids[index])

        return (image_representation, caption_tokenized_id)

    def __len__(self) -> int:
        """
        Return the size of this dataset.

        return: Size of this dataset.
        """
        return self.length

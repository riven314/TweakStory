import functools
import json
import os
import logging
import pathlib
from typing import Any, Callable, Dict, List

import torchvision
import yaml

import torch


class Loader(yaml.SafeLoader):
    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)


# enable PyYAML to handle "!include"
Loader.add_constructor('!include', Loader.include)


class EnrichedEncoder(json.JSONEncoder):

    def default(self, obj: Any) -> str:
        if isinstance(obj, pathlib.Path):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


def log_init(constructor: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(constructor)
    def wrapper(*args: Any, **kwargs: Dict[str, Any]) -> Any:
        self = args[0]
        self.logger = logging.getLogger(self.__class__.__name__)
        arguments = json.dumps(kwargs, cls=EnrichedEncoder, indent=2)
        self.logger.info(
            f"Initializing: {self.__class__.__name__} with:\n{arguments}\n"
        )
        constructor(*args, **kwargs)
    return wrapper


def log_run(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Dict[str, Any]) -> Any:
        self = args[0]
        self.logger.info(
            f"Starting {self.__class__.__name__}"
        )
        return func(*args, **kwargs)

    return wrapper


def create_pad_collate(
    pad_token_id: int
) -> Callable[[List[Any]], Any]:
    def pad_collate(
        batch: List[Any]
    ) -> Any:
        images = []
        captions = []
        for entry in batch:
            images.append(entry[0])
            captions.append(entry[1])
        images_tensor = torch.stack(images)
        captions_padded = torch.nn.utils.rnn.pad_sequence(
            captions, batch_first=True, padding_value=pad_token_id
        )

        return (images_tensor, captions_padded)

    return pad_collate


def get_image_transformations(
    config: Dict[str, Any]
) -> torchvision.transforms.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=config["image_size"]
            ),
            torchvision.transforms.CenterCrop(
                size=config["crop_size"]
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=config["image_normalization_mean"],
                std=config["image_normalization_std"],
            )
        ]
    )

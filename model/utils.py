import functools
import json
import logging
import pathlib
from typing import Any, Callable, Dict, Tuple

import torch


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
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    def pad_collate(
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

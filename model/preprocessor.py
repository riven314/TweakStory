import json
import logging
import logging.config
import re
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Pattern

import bpemb
import emoji
import h5py
import numpy
import requests
from tqdm import tqdm

from utils import log_init, log_run


class PipelineStep(ABC):

    @abstractmethod
    def run(self) -> None:
        pass


class IGPreprocessor(PipelineStep):

    @log_init
    def __init__(
        self,
        *,
        config: Dict[str, Any],
        force_update: bool = False
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config[self.__class__.__name__]
        self.force_update = force_update

        raw_data_url = self.config["raw_data_url"]
        raw_data_path = Path(self.config["raw_data_path"])
        raw_data_directory = self.config["raw_data_directory"]
        raw_data_group_names = self.config["raw_data_group_names"]
        image_directory = raw_data_path.parent / raw_data_directory / "images"
        json_directory = raw_data_path.parent / raw_data_directory / "json"
        hdf5_path = raw_data_path.parent / (raw_data_directory + ".hdf5")

        self.loader = IGLoader(
            raw_data_url=raw_data_url,
            raw_data_path=raw_data_path,
            force_update=self.force_update
        )

        self.hdf = IGHDF(
            hdf5_path=hdf5_path,
            raw_data_directory=self.config["raw_data_directory"],
            image_directory=image_directory,
            json_directory=json_directory,
            raw_data_group_names=raw_data_group_names,
            force_update=self.force_update
        )

        self.cleaner = IGCaptionCleaner(
            hdf5_path=hdf5_path,
            raw_data_directory=raw_data_directory,
            image_directory=image_directory,
            raw_data_group_names=raw_data_group_names

        )

        self.tokenizer = BPTokenizer(
            hdf5_path=hdf5_path,
            raw_data_group_names=raw_data_group_names
        )

    @log_run
    def run(self) -> None:
        self.logger.info(f"Starting {self.__class__.__name__}")
        self.loader.run()
        self.hdf.run()
        self.cleaner.run()
        self.tokenizer.run()


class IGLoader(PipelineStep):

    @log_init
    def __init__(
        self,
        *,
        raw_data_url: str,
        raw_data_path: Path,
        force_update: bool = False
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.raw_data_url = raw_data_url
        self.raw_data_path = raw_data_path
        self.force_update = force_update

    @log_run
    def run(self) -> None:
        self.download_raw_data()
        self.unpack_raw_data()

    def download_raw_data(self) -> None:
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)

        if (
            self.raw_data_path.is_file() and
            not self.force_update
        ):
            self.logger.info(
                "Cached version of raw data already exists. Skipping download."
            )
        else:
            self.logger.info(
                "Downloading raw data from " +
                self.raw_data_url +
                "to " +
                str(self.raw_data_path)
            )
            response = requests.get(self.raw_data_url, stream=True)

            progress_bar = tqdm(
                total=int(response.headers.get("content-length", 0)),
                unit="iB",
                unit_scale=True
            )
            with open(self.raw_data_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(chunk))
                    if chunk:
                        f.write(chunk)

    def unpack_raw_data(self) -> None:
        with tarfile.open(self.raw_data_path) as f:
            directory = f.getnames()[0]

            if (
                (self.raw_data_path.parent / directory).exists() and
                not self.force_update
            ):
                self.logger.info(
                    "Cached version of raw data already exists. " +
                    "Skipping extraction."
                )

            else:
                self.logger.info(
                    "Extracting raw data from " +
                    str(self.raw_data_path) +
                    " to " +
                    str(self.raw_data_path.parent)
                )
                f.extractall(self.raw_data_path.parent)


class IGHDF(PipelineStep):

    @log_init
    def __init__(
        self,
        *,
        hdf5_path: Path,
        raw_data_directory: str,
        image_directory: Path,
        json_directory: Path,
        raw_data_group_names: Dict[str, str],
        force_update: bool = False
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hdf5_path = hdf5_path
        self.raw_data_directory = raw_data_directory
        self.image_directory = image_directory
        self.json_directory = json_directory
        self.raw_data_group_names = raw_data_group_names
        self.force_update = force_update

    def run(self) -> None:
        cached = True

        if self.force_update and self.hdf5_path.is_file():
            cached = False
            with h5py.File(self.hdf5_path, "w") as hdf5_store:
                del hdf5_store

        if self.hdf5_path.is_file() and cached:
            self.logger.info(
                "Cached version of hdf5 store already exists. Skipping step."
            )

        else:
            self.logger.info(f"Creating hdf5 store at {self.hdf5_path}")

            with h5py.File(self.hdf5_path, "a") as hdf5_store:
                for key, value in self.raw_data_group_names.items():
                    if value in hdf5_store.keys():
                        hdf5_store = hdf5_store[value]
                    else:
                        hdf5_group = hdf5_store.create_group(
                            value,
                            track_order=True
                        )
                    caption_id = []
                    caption_raw = []
                    with open(self.json_directory / key) as f:
                        raw_json = json.load(f)

                    for user, entry in raw_json.items():
                        for post_id, post in entry.items():
                            caption_id.append(user + "_@_" + post_id)
                            caption_raw.append(post["caption"])
                    hdf5_group.create_dataset(
                        "caption_id",
                        data=numpy.array(
                            caption_id,
                            dtype=h5py.string_dtype(encoding="utf-8")
                        )
                    )
                    hdf5_group.create_dataset(
                        "caption_raw",
                        data=numpy.array(
                            caption_raw,
                            dtype=h5py.string_dtype(encoding="utf-8")
                        )
                    )


class IGCaptionCleaner(PipelineStep):

    @log_init
    def __init__(
        self,
        *,
        hdf5_path: Path,
        raw_data_group_names: Dict[str, str],
        image_directory: Path,
        raw_data_directory: str,
        force_update: bool = False

    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hdf5_path = hdf5_path
        self.raw_data_directory = raw_data_directory
        self.image_directory = image_directory
        self.raw_data_group_names = raw_data_group_names
        self.force_update = force_update

        self.username_regex = re.compile(r"@([A-Za-z0-9_]|\.[A-Za-z0-9_])+")
        self.username_placeholder = "@username "
        self.whitespace_regex = re.compile(r"\s+")
        self.whitespace_placeholder = " "

    def run(self) -> None:
        # TODO: check whether cache exists
        self.logger.info(f"Cleaning caption data.")
        for hdf5_group in self.raw_data_group_names.values():
            self.anonymize_usernames(
                input_hdf5_group=hdf5_group,
                input_hdf5_dataset="caption_raw",
                output_hdf5_group=hdf5_group,
                output_hdf5_dataset="caption_cleaned"
            )

            self.encode_emoji(
                input_hdf5_group=hdf5_group,
                input_hdf5_dataset="caption_cleaned",
                output_hdf5_group=hdf5_group,
                output_hdf5_dataset="caption_cleaned"
            )

            # normalize_whitespace needs to be the last step
            self.normalize_whitespace(
                input_hdf5_group=hdf5_group,
                input_hdf5_dataset="caption_cleaned",
                output_hdf5_group=hdf5_group,
                output_hdf5_dataset="caption_cleaned"
            )

    def regex_substitution(
        self,
        regex: Pattern,
        substitution: str,
        input_hdf5_group: str,
        input_hdf5_dataset: str,
        output_hdf5_group: str,
        output_hdf5_dataset: str
    ) -> None:
        with h5py.File(self.hdf5_path, "a") as hdf5_store:
            self.logger.info(hdf5_store.keys())
            captions = numpy.array(
                hdf5_store.get(
                    input_hdf5_group
                ).get(
                    input_hdf5_dataset
                )
            )

            captions_cleaned = []

            for caption in captions:
                caption_cleaned = re.sub(
                    regex,
                    substitution,
                    caption
                )

                captions_cleaned.append(caption_cleaned)

            output_group = hdf5_store.require_group(output_hdf5_group)
            if output_hdf5_dataset in output_group.keys():
                del output_group[output_hdf5_dataset]
            output_group.create_dataset(
                output_hdf5_dataset,
                data=numpy.array(
                    captions_cleaned,
                    dtype=h5py.string_dtype(encoding="utf-8")
                )
            )

    def anonymize_usernames(
        self,
        input_hdf5_group: str,
        input_hdf5_dataset: str,
        output_hdf5_group: str,
        output_hdf5_dataset: str
    ) -> None:
        self.logger.info(
            "Anonymizing usernames. Data transfer:\n" +
            f"\"{input_hdf5_group}/{input_hdf5_dataset}\" -> " +
            f"\"{output_hdf5_group}/{output_hdf5_dataset}\"."
        )

        self.regex_substitution(
            self.username_regex,
            self.username_placeholder,
            input_hdf5_group,
            input_hdf5_dataset,
            output_hdf5_group,
            output_hdf5_dataset
        )

    def normalize_whitespace(
        self,
        input_hdf5_group: str,
        input_hdf5_dataset: str,
        output_hdf5_group: str,
        output_hdf5_dataset: str
    ) -> None:
        self.logger.info(
            "Normalizing whitespace. Data transfer:\n" +
            f"\"{input_hdf5_group}/{input_hdf5_dataset}\" -> " +
            f"\"{output_hdf5_group}/{output_hdf5_dataset}\"."
        )

        self.regex_substitution(
            self.whitespace_regex,
            self.whitespace_placeholder,
            input_hdf5_group,
            input_hdf5_dataset,
            output_hdf5_group,
            output_hdf5_dataset
        )

        with h5py.File(self.hdf5_path, "a") as hdf5_store:
            self.logger.info(hdf5_store.keys())
            captions = numpy.array(
                hdf5_store.get(
                    input_hdf5_group
                ).get(
                    input_hdf5_dataset
                )
            )

            captions_cleaned = []

            for caption in captions:
                caption_cleaned = caption.strip()
                captions_cleaned.append(caption_cleaned)

            output_group = hdf5_store.require_group(output_hdf5_group)
            if output_hdf5_dataset in output_group.keys():
                del output_group[output_hdf5_dataset]
            output_group.create_dataset(
                output_hdf5_dataset,
                data=numpy.array(
                    captions_cleaned,
                    dtype=h5py.string_dtype(encoding="utf-8")
                )
            )

    def encode_emoji(
        self,
        input_hdf5_group: str,
        input_hdf5_dataset: str,
        output_hdf5_group: str,
        output_hdf5_dataset: str
    ) -> None:
        with h5py.File(self.hdf5_path, "a") as hdf5_store:
            self.logger.info(hdf5_store.keys())
            captions = numpy.array(
                hdf5_store.get(
                    input_hdf5_group
                ).get(
                    input_hdf5_dataset
                )
            )

            captions_cleaned = []

            for caption in captions:
                caption_cleaned = emoji.demojize(
                    caption,
                    delimiters=(" :", ": ")
                )
                captions_cleaned.append(caption_cleaned)

            output_group = hdf5_store.require_group(output_hdf5_group)
            if output_hdf5_dataset in output_group.keys():
                del output_group[output_hdf5_dataset]
            output_group.create_dataset(
                output_hdf5_dataset,
                data=numpy.array(
                    captions_cleaned,
                    dtype=h5py.string_dtype(encoding="utf-8")
                )
            )


class BPTokenizer(PipelineStep):

    @log_init
    def __init__(
        self,
        hdf5_path: Path,
        raw_data_group_names: Dict[str, str],
        language: str = "en",
        vocabulary_size: int = 100_000,
        embedding_dimensionality: int = 300,
        force_update: bool = False
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hdf5_path = hdf5_path
        self.raw_data_group_names = raw_data_group_names
        self.language = language
        self.vocabulary_size = vocabulary_size
        self.embedding_dimensionality = embedding_dimensionality
        self.tokenizer = bpemb.BPEmb(
            lang=self.language,
            vs=self.vocabulary_size,
            dim=self.embedding_dimensionality
        )
        self.force_update = force_update

    def run(self) -> None:
        # TODO: check whether cache exists
        self.logger.info(f"Tokenizing caption data.")

        with h5py.File(self.hdf5_path, "a") as hdf5_store:
            for hdf5_group_name in self.raw_data_group_names.values():
                hdf5_group = hdf5_store.get(
                    hdf5_group_name
                )
                captions = numpy.array(
                    hdf5_group["caption_cleaned"]
                )

                captions_tokenized = []
                captions_tokenized_id = []

                for caption in captions:
                    caption_tokenized = self.tokenizer.encode(caption)
                    caption_tokenized_id = self.tokenizer.encode_ids(caption)
                    captions_tokenized.append(caption_tokenized)
                    captions_tokenized_id.append(caption_tokenized_id)

                if "caption_cleaned_tokenized" in hdf5_group.keys():
                    del hdf5_group["caption_cleaned_tokenized"]
                if "caption_cleaned_tokenized_id" in hdf5_group.keys():
                    del hdf5_group["caption_cleaned_tokenized_id"]

                hdf5_group.create_dataset(
                    "caption_cleaned_tokenized",
                    data=numpy.array(
                        captions_tokenized,
                        dtype=h5py.string_dtype(encoding="utf-8")
                    )
                )
                token_id_dataset = hdf5_group.create_dataset(
                    "caption_cleaned_tokenized_id",
                    shape=(len(captions_tokenized_id),),
                    dtype=h5py.vlen_dtype(numpy.dtype("int32"))
                )
                token_id_dataset[...] = captions_tokenized_id

import json
import logging
import logging.config
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy
import requests
from tqdm import tqdm


class PipelineStep(ABC):

    @abstractmethod
    def run(self) -> None:
        pass


class IGPreprocessor(PipelineStep):

    def __init__(
        self,
        config: Dict[str, Any],
        force_update: bool = False
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config[self.__class__.__name__]
        self.force_update = force_update

        self.logger.info(
            f"Initializing {self.__class__.__name__} with config\n" +
            json.dumps(self.config, indent=4) +
            f",\nforce_update = {self.force_update}"
        )
        self.loader = IGLoader(
            raw_data_url=self.config["raw_data_url"],
            raw_data_path=Path(self.config["raw_data_path"]),
            force_update=self.force_update
        )

        self.hdf = IGHDF(
            raw_data_directory=self.config["raw_data_directory"],
            image_directory=(
                Path(
                    self.config["raw_data_path"]
                ).parent /
                self.config["raw_data_directory"] /
                "images"
            ),
            json_directory=(
                Path(
                    self.config["raw_data_path"]
                ).parent /
                self.config["raw_data_directory"] /
                "json"
            ),
            raw_data_group_names=self.config["raw_data_group_names"],
            force_update=self.force_update
        )

    def run(self) -> None:
        self.logger.info(f"Starting {self.__class__.__name__}")
        self.loader.run()
        self.hdf.run()


class IGLoader(PipelineStep):

    def __init__(
        self,
        raw_data_url: str,
        raw_data_path: Path,
        force_update: bool = False
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.raw_data_url = raw_data_url
        self.raw_data_path = raw_data_path
        self.force_update = force_update

        self.logger.info(
            f"Initializing {self.__class__.__name__} with\n" +
            f"raw_data_url = {self.raw_data_url},\n" +
            f"raw_data_path = {self.raw_data_path},\n" +
            f"force_update = {self.force_update}"
        )

    def run(self) -> None:
        self.logger.info(f"Starting {self.__class__.__name__}")
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
                total=int(response.headers.get('content-length', 0)),
                unit='iB',
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

    def __init__(
        self,
        raw_data_directory: str,
        image_directory: Path,
        json_directory: Path,
        raw_data_group_names: Dict[str, str],
        force_update: bool = False
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.raw_data_directory = raw_data_directory
        self.image_directory = image_directory
        self.json_directory = json_directory
        self.raw_data_group_names = raw_data_group_names
        self.force_update = force_update

        self.logger.info(
            f"Initializing {self.__class__.__name__} with\n" +
            f"image_directory = {self.image_directory},\n" +
            f"json_directory = {self.json_directory},\n" +
            f"force_update = {self.force_update}"
        )

    def run(self) -> None:
        hdf5_path = (
            Path(self.image_directory.parent) /
            (self.raw_data_directory + ".hdf5")
        )

        if self.force_update:
            hdf5_path.unlink(missing_ok=True)

        if hdf5_path.is_file():
            self.logger.info(
                "Cached version of hdf5 store already exists. Skipping step."
            )

        else:
            self.logger.info(f"Creating hdf5 store at {hdf5_path}")

            with h5py.File(hdf5_path, "w") as hdf5:
                for key, value in self.raw_data_group_names.items():
                    hdf5_group = hdf5.create_group(value)
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
                            dtype=h5py.string_dtype(encoding='utf-8')
                        )
                    )
                    hdf5_group.create_dataset(
                        "caption_raw",
                        data=numpy.array(
                            caption_raw,
                            dtype=h5py.string_dtype(encoding='utf-8')
                        )
                    )

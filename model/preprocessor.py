from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict
from tqdm import tqdm
import json
import logging
import logging.config
import requests
import tarfile


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
        self.force_update = False

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
            )
        )

    def run(self) -> None:
        self.logger.info(f"Starting {self.__class__.__name__}")
        self.loader.run()


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
        image_directory: Path,
        json_directory: Path,
        force_update: bool = False
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.image_directory = image_directory
        self.json_directory = json_directory
        self.force_update = force_update

        self.logger.info(
            f"Initializing {self.__class__.__name__} with\n" +
            f"image_directory = {self.image_directory},\n" +
            f"json_directory = {self.json_directory},\n" +
            f"force_update = {self.force_update}"
        )

    def run(self) -> None:
        pass

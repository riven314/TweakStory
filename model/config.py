from pathlib import Path

import yaml

from utils import Loader

with open(Path("./config/config.yaml")) as f:
    config = yaml.load(f, Loader=Loader)

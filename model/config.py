from pathlib import Path
import yaml

with open(Path("./config/config.yaml")) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

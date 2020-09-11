import argparse
import logging

from config import config
from preprocessor import IGPreprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_update", type=bool, default=False)
    args = parser.parse_args()
    logging.config.dictConfig(config["logger"])
    print("bla:", args.force_update)
    preprocessor = IGPreprocessor(
        config,
        force_update=args.force_update
    )

    preprocessor.run()

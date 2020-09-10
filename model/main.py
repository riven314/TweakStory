from config import config
from preprocessor import IGPreprocessor
import logging

if __name__ == "__main__":
    logging.config.dictConfig(config["logger"])
    preprocessor = IGPreprocessor(config)

    preprocessor.run()

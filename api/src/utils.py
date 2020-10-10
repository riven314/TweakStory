import os
from io import BytesIO
from base64 import b64decode

import numpy as np
from PIL import Image

from src.common import read_json, read_yaml


def decode_b64_image_string(b64_img_str):
    bytes_img = BytesIO(b64decode(b64_img_str.encode('ascii')))
    np_img = np.array(Image.open(bytes_img).convert('RGB'))
    return np_img

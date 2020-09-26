import os
import json
import yaml
from io import BytesIO
from base64 import b64decode

import numpy as np
from PIL import Image


def read_json(json_path):
    assert json_path, f'{json_path} not exist'
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


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


def read_yaml(yaml_path):
    assert yaml_path, f'{yaml_path} not exist'
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader = Loader)
    return data


def decode_b64_image_string(b64_img_str):
    bytes_img = BytesIO(b64decode(b64_img_str.encode('ascii')))
    np_img = np.array(Image.open(bytes_img).convert('RGB'))
    return np_img

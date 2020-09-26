import os
import yaml

import numpy as np
from PIL import Image


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


def open_image(img_buffer, demo_flag):
    img = Image.open(img_buffer).convert('RGB')
    img = np.array(img).astype(np.uint8)
    
    if len(img) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis = 2)
    return img
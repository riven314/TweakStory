import os
import sys
import yaml
from io import BytesIO

import cv2
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


def open_image(img_bytes, demo_flag):
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    img = np.array(img).astype(np.uint8)
    
    if len(img) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis = 2)
    return img


def query_file_kb(file_bytes):
    """ express file size in KB """
    file_kb = sys.getsizeof(file_bytes) * 0.001
    return file_kb


def resize_to_aspect_ratio_and_limit_length(np_img, max_length):
    h, w, _ = np_img.shape
    aspect_ratio = w / h

    if w >= h:
        tgt_w = int(max_length)
        tgt_h = int(tgt_w / aspect_ratio)
    else:
        tgt_h = int(max_length)
        tgt_w = int(tgt_h * aspect_ratio)
    np_img = cv2.resize(np_img, (tgt_w, tgt_h))
    return np_img


def resize_img_bytes(img_bytes, max_length):
    """ resize the bytes of a bigger image to the bytes of a smaller image """
    np_img = np.array(Image.open(BytesIO(img_bytes)))
    resized_np_img = resize_to_aspect_ratio_and_limit_length(np_img, max_length)
    resized_img = Image.fromarray(resized_np_img)
    bytesio = BytesIO()
    resized_img.save(bytesio, format = 'JPEG')
    resized_img_bytes = bytesio.getvalue()
    return resized_img_bytes

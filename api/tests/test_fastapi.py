import os
import json
import requests
from base64 import b64encode

import streamlit as st


def test_fastapi_endpoint():
    REQUEST_SUCCESS = 200
    URL = 'http://127.0.0.1:8080/inference'
    IMAGE_PATH = './demo/demo_img1.jpg'
    assert os.path.isfile(IMAGE_PATH)

    bytes_img = b64encode(open(IMAGE_PATH, 'rb').read())
    b64_str_img = bytes_img.decode("ascii")

    for sentence_class in [0, 1, 2]:
        for emoji_class in [0, 1]:
            body = dict(sentence_class = sentence_class,
                        emoji_class = emoji_class,
                        b64_img_str = b64_str_img)

            res = requests.post(url = URL, data = json.dumps(body))
            assert res.status_code == REQUEST_SUCCESS
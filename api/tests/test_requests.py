import os
from base64 import b64encode

from tests.config import *


def _stringify_b64_encoded_image(img_path):
    assert os.path.isfile(img_path)

    bytes_img = b64encode(open(img_path, 'rb').read())
    b64_img_str = bytes_img.decode('ascii')
    return b64_img_str


def test_fastapi_endpoint(test_client):
    b64_img_str = _stringify_b64_encoded_image(IMAGE_PATH)

    for sentence_class in SENTENCE_CLASSES:
        for emoji_class in EMOJI_CLASSES:
            body = dict(
                    sentence_class = sentence_class,
                    emoji_class = emoji_class,
                    b64_img_str = b64_img_str
                )

            res = test_client.post(REQUEST_ROUTE, json = body)
            
            assert res.status_code == SUCCESS_CODE

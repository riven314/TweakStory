# uvicorn {script name}:{FastAPI instance name} --port 8080
from io import BytesIO
from base64 import b64decode
import time
import logging

import cv2
import torch
import numpy as np
from PIL import Image
from easydict import EasyDict as edict

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from src.app_utils import *
from src.constants import *
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()


app_cfg = edict(read_yaml(CONFIG_FILE))
model_cfg = edict(app_cfg.model_config)
encoder, decoder = setup_models(model_cfg, is_cuda = app_cfg.is_cuda)
word_map = read_json(model_cfg.word_map_file)
rev_word_map = {v: k for k, v in word_map.items()}
tokenizer = setup_tokenizer(word_map)


def model_inference(np_img, sentence_class, emoji_class):
    resized_img = cv2.resize(np_img, (app_cfg.img_resize, app_cfg.img_resize))
    tensor_img = tfms_image(resized_img)
    
    # caption not yet emojized (e.g. :hugging_face:)
    caption, pred_ids, _ = output_caption(
        encoder, decoder, tensor_img, 
        word_map, rev_word_map, tokenizer,
        sentence_class, emoji_class,
        beam_size = app_cfg.beam_size
    )
    return caption


def decode_b64_image_string(b64_img_str):
    bytes_img = BytesIO(b64decode(b64_img_str.encode('ascii')))
    np_img = np.array(Image.open(bytes_img).convert('RGB'))
    return np_img


app = FastAPI()

class UserControl(BaseModel):
    sentence_class: int
    emoji_class: int
    b64_img_str: str


@app.post("/inference")
def get_model_caption(usr_ctrl: UserControl):
    start = time.time()

    sentence_class = usr_ctrl.sentence_class
    emoji_class = usr_ctrl.emoji_class
    np_img = decode_b64_image_string(usr_ctrl.b64_img_str)

    caption = model_inference(np_img, sentence_class, emoji_class)
    
    t = time.time() - start
    logger.info(f'inference complete: {t:.5f} s')
    logger.info(f'generated caption: {caption}')
    return {'output': caption}  
    

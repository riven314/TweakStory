import os
import time
import logging

import cv2
import torch
import emoji
import numpy as np
from easydict import EasyDict as edict

from fastapi import FastAPI

from src.app_utils import *
from src.constants import *


app = FastAPI()

app_cfg = edict(read_yaml(CONFIG_FILE))
model_cfg = edict(app_cfg.model_config)

encoder, decoder = setup_models(model_cfg, is_cuda = app_cfg.is_cuda)
word_map = read_json(model_cfg.word_map_file)
rev_word_map = {v: k for k, v in word_map.items()}
tokenizer = setup_tokenizer(word_map)


def model_inference(img_buffer, sentence_class_str, emoji_class_str):
    sentence_class = SENTENCE_CLASS_MAP[sentence_class_str]
    emoji_class = EMOJI_CLASS_MAP[emoji_class_str]

    img_fn, demo_flag = (img_buffer, False) if img_buffer is not None else (app_cfg.demo_image, True)
    np_img = open_image(io_img, demo_flag)
    resized_img = cv2.resize(np_img, (app_cfg.img_resize, app_cfg.img_resize))
    tensor_img = tfms_image(resized_img)
    
    # caption not yet emojized
    caption, pred_ids, _ = output_caption(
        encoder, decoder, tensor_img, 
        word_map, rev_word_map, tokenizer,
        sentence_class, emoji_class,
        beam_size = app_cfg.beam_size
    )
    return caption

    
    

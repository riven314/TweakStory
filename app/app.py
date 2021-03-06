import os
import time

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
import streamlit as st
from PIL import Image

from src.app_utils import *
from src.constants import *

app_cfg = edict(read_yaml(CONFIG_FILE))
model_cfg = edict(app_cfg.model_config)
st.beta_set_page_config(page_title = 'Tweak Story', page_icon = app_cfg.page_icon)
st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(show_spinner = False, allow_output_mutation = True)
def get_models():
    encoder, decoder = setup_models(model_cfg, is_cuda = app_cfg.is_cuda)
    print('model received')
    return encoder, decoder


@st.cache(show_spinner = False, allow_output_mutation = True)
def get_tokenizer():
    tokenizer = setup_tokenizer(word_map)
    print('tokenizer received')
    return tokenizer


@st.cache(show_spinner = False, allow_output_mutation = True)
def get_word_maps():
    word_map_file = model_cfg.word_map_file
    word_map = read_json(word_map_file)
    rev_word_map = {v: k for k, v in word_map.items()}
    print('word map received')
    return word_map, rev_word_map


# preset models and tokenizer etc.
encoder, decoder = get_models()
device = torch.device('cuda' if next(encoder.parameters()).is_cuda else 'cpu')
word_map, rev_word_map = get_word_maps()
tokenizer = get_tokenizer() # set tokenizer only after word_map


# user input on sidebar
st.sidebar.header('Step 1: Upload Image')
img_buffer = st.sidebar.file_uploader(
    '',
    type = ['png', 'jpg', 'jpeg']
)
st.sidebar.text(" \n")
st.sidebar.text(" \n")

st.sidebar.header('Step 2: Select Your Flavors')
st.sidebar.text(" \n")
len_choices = tuple([k for k in SENTENCE_CLASS_MAP.keys()])
sentence_class = st.sidebar.selectbox( '', len_choices)
emoji_choices = tuple([k for k in EMOJI_CLASS_MAP.keys()])
emoji_class = st.sidebar.selectbox('', emoji_choices)
st.sidebar.text(" \n")
st.sidebar.text(" \n")

st.sidebar.header('Step 3: Generate Caption!')
st.sidebar.text(" \n")
is_run = st.sidebar.button('RUN')


# propagate user input to model run
img_fn, demo_flag = (img_buffer, False) if img_buffer is not None else (app_cfg.demo_image, True)
np_img = open_image(img_fn, demo_flag)
resized_img = cv2.resize(np_img, (app_cfg.img_resize, app_cfg.img_resize))
caption = emoji.emojize(f"{DEFAULT_PADDING} :backhand_index_pointing_left: {PADDING_CODE} Press RUN Button to Generate Caption")

if is_run:
    tensor_img = tfms_image(resized_img)
    sentence_class = SENTENCE_CLASS_MAP[sentence_class]
    emoji_class = EMOJI_CLASS_MAP[emoji_class]

    caption, pred_ids, _ = output_caption(
        encoder, decoder, tensor_img, 
        word_map, rev_word_map, tokenizer, 
        sentence_class, emoji_class, 
        beam_size = app_cfg.beam_size
    )
    caption = f'{DEFAULT_PADDING} <b>CAPTION</b> {caption}'


# display image (preserve aspect ratio)
h, w, _ = np_img.shape
aspect_ratio = w / h

if w >= h:
    tgt_w = int(app_cfg.img_display)
    tgt_h = int(tgt_w / aspect_ratio)
else:
    tgt_h = int(app_cfg.img_display)
    tgt_w = int(tgt_h * aspect_ratio)
np_img = cv2.resize(np_img, (tgt_w, tgt_h))

st.image(Image.fromarray(np_img), use_column_width = False)
st.write('')
st.markdown(f""" <h3> {caption} </h3> """, 
            unsafe_allow_html = True)


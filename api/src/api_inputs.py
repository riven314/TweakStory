import cv2
from easydict import EasyDict as edict

from src.utils import read_yaml, read_json
from src.infer_utils import tfms_image, output_caption, setup_models, setup_tokenizer

CONFIG_FILE = './config/api_config.yaml' 


# set up model-specific dependencies
app_cfg = edict(read_yaml(CONFIG_FILE))
model_cfg = edict(app_cfg.model_config)
encoder, decoder = setup_models(model_cfg, is_cuda = False)
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
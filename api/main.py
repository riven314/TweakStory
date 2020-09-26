# uvicorn {script name}:{FastAPI instance name} --port 8080
import time
import logging

from fastapi import FastAPI
from pydantic import BaseModel

from src.utils import decode_b64_image_string
from src.api_inputs import model_inference

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()


# set up FastAPI server
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
    

import cv2
import json
import yaml
import logging

from PIL import Image

import matplotlib.pyplot as plt


# ===== READ/WRITE DATA FORMAT =====
def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML: {e}")


# ===== STRING-DICT CONVERSION =====
def json_to_str(payload: dict) -> str:
    return json.dumps(payload)

def str_to_json(payload: str) -> dict:
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")


# ===== LOGGING SYSTEM =====
def set_logger(level: int = logging.DEBUG, file_path: str = None) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s] - %(levelname)7s --- %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if file_path:
        try:
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error creating log file: {e}")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger  


# ===== PLOT KEY-FRAMES =====
def get_frame(video, frame_id):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    _, frame = video.read()
    
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def fml_frames(video, shot): # first, mid, last frame
    first_id, last_id = shot
    mid_id = (last_id - first_id) // 2 + first_id
    
    first_frame = get_frame(video, first_id)
    mid_frame = get_frame(video, mid_id)
    last_frame = get_frame(video, last_id)
    
    return [(first_frame, first_id), (mid_frame, mid_id), (last_frame, last_id)]

def show_fml_keyframes(keyframes):
    len_keyframes =  len(keyframes)
    
    plt.figure(figsize=(10, 5))
    for i in range(len_keyframes):
        frame = keyframes[i]
        plt.subplot(1, len_keyframes, i+1)
        plt.imshow(frame[0])
        plt.title(f'Frame {frame[1]}')
        
    plt.show()

def show_lmske_keyframes(video, keyframes):
    len_keyframes =  len(keyframes)
    
    plt.figure(figsize=(10, 5))
    for i in range(len_keyframes):
        frame_i = keyframes[i]
        plt.subplot(1, len_keyframes, i+1)
        plt.imshow(get_frame(video, frame_i))
        plt.title(f'Frame {frame_i}')
        
    plt.show()

    
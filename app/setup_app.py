import os
import glob
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch

from routes import base_route

from app.src.helpers import read_yaml, set_logger
from app.src.embedding.blip_ import BlipTool
from app.src.translation import GoogleTranslator
from app.src.vector_store import VectorStore


config = read_yaml('./config.yaml')
logging = set_logger()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@asynccontextmanager
async def lifespan(app: FastAPI): 
    
    logging.info("[INFO] Setup lifespan ...")

    logging.info("[INFO] Setup Paths ...")
    env_dir = config['environment']
    lst_keyframes = glob.glob(os.path.join(env_dir['root'], 
                            f'{env_dir['lst_keyframes']['path']}', 
                            f'*{env_dir['lst_keyframes']['format']}'))
    lst_keyframes.sort()

    id2img_fps = dict()
    for i, img_path in enumerate(lst_keyframes):
        id2img_fps[i] = img_path

    # Setup Translator
    logging.info("[INFO] Setup Translator ...")
    app.state.translator = GoogleTranslator()
    
    
    # Setup Embedding Model
    logging.info("[INFO] Setup Embedding Model ...")
    embedding_model_cfg = config['blip']
    embedding_model = BlipTool(model_name=embedding_model_cfg['model_name'])
    app.state.embedding_model = embedding_model


    # Setup Vector Store
    logging.info("[INFO] Setup Vector Store ...")
    root_features = os.path.join(env_dir['root'], env_dir['features'])
    bin_name = os.path.join(root_features, embedding_model_cfg['bin_name'] + '.bin')
    bin_file= os.path.join(root_features, f'{bin_name}.bin')
    
    vector_store = VectorStore(bin_file, id2img_fps, device, embedding_model)
    app.state.vector_store = vector_store

    yield
    
    app.state.translator.close()
    app.state.embedding_model.close()
    app.state.vector_store.close()
    logging.info("[INFO] Clean up lifespan ...")
    

def setup_app(lifspan=None) -> FastAPI: 
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
    CORSMiddleware, # https://fastapi.tiangolo.com/tutorial/cors/
    allow_origins=['*'], # wildcard to allow all, more here - https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
    allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
    allow_methods=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
    allow_headers=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
)

    app.include_router(base_route)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    return app

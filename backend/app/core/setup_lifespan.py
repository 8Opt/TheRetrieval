import os
import glob
from contextlib import asynccontextmanager

from core.config import Config
from core.logger import set_logger

from utils.helpers import ignore_warning, get_to_root

get_to_root()
ignore_warning()

config = Config()
logging = set_logger()

@asynccontextmanager
async def lifespan(app): 
    
    logging.info("[INFO] Setup lifespan ...")

    logging.info("[INFO] Setup Paths ...")
    env_dir = config.environment
    lst_keyframes = glob.glob(os.path.join(env_dir.root, 
                            f"{env_dir.lst_keyframes['path']}", 
                            f"*{env_dir.lst_keyframes['format']}"))
    lst_keyframes.sort()

    id2img_fps = dict()
    for i, img_path in enumerate(lst_keyframes):
        id2img_fps[i] = img_path

    # Setup Translator
    logging.info("[INFO] Setup Translator ...")
    app.state.translator = config.translator
    
    
    # Setup Embedding Model
    logging.info("[INFO] Setup Embedding Model ...")
    app.state.embedding_model = config.embedding_model


    # Setup Vector Store
    logging.info("[INFO] Setup Vector Store ...")
    root_features = os.path.join(env_dir.root, env_dir.features)
    bin_file= os.path.join(root_features, f'{config.embedding_model.bin_name}.bin')
    
    vector_store = config.vector_store(env_dir.root, bin_file, id2img_fps, config.device, config.embedding_model)
    app.state.vector_store = vector_store

    yield
    
    logging.info("[INFO] Clean up lifespan ...")

    del app.state.translator
    del app.state.embedding_model
    del app.state.vector_store    

from dataclasses import dataclass

import torch

from models.translation import GoogleTranslator
from models.embedding.blip_ import BlipTool
from services.vector_store import VectorStore
from utils.helpers import ignore_warning


ignore_warning()

@dataclass
class Environment:
    device = "cuda"
    root = "../db"                  # Need to go outside `/backend`
    features = "features"
    lst_keyframes = {
        'path': "s_optimized_keyframes", 
        "format": ".webp"
    }
    vid_url = "vid_url.json"


@dataclass
class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    environment = Environment()
    translator = GoogleTranslator()
    embedding_model = BlipTool(device=device)

    # TODO: Make this fix based on settings on this file, not in `setup_lifespan`
    vector_store = VectorStore

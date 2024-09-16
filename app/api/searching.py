import os
import requests
from json import loads
from dotenv import load_dotenv

load_dotenv()


def search_by_text(query, top_k:int, vector_store): 
    scores, idx_image, infos_query, image_paths = vector_store.text_search(query, top_k)
    return {
        'scores': scores.tolist(),
        'idx_image': idx_image.tolist(),
        'infos_query': infos_query, 
        'image_paths': image_paths
    }
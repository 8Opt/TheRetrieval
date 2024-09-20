from dotenv import load_dotenv

load_dotenv()


def search_by_text(query:str, top_k:int, vector_store): 
    scores, idx_image, infos_query, image_paths = vector_store.text_search(query, top_k)
    
    infos_query = [info.split('/')[-1] for info in infos_query]
    image_paths = [image_path.split('/')[-1] for image_path in image_paths]

    return {
        'scores': scores.tolist(),
        'idx_image': idx_image.tolist(),
        'infos_query': infos_query, 
        'image_paths': image_paths
    }
import numpy as np
import torch
import faiss
from PIL import Image
import requests

class VectorStore:
    """
    A class for storing and searching vectors using FAISS.

    Attributes:
        root (str): The root directory of the vector store.
        index (faiss.Index): The FAISS index.
        id2img_fps (dict): A dictionary mapping IDs to image file paths.
        device (str): The device to use for computations (e.g., 'cuda' or 'cpu').
        embedding_model: The embedding model used for feature extraction.
    """

    def __init__(self, root: str, bin_file: str, id2img_fps: dict, device: str, embedding_model):
        """
        Initializes the VectorStore instance.

        Args:
            root (str): The root directory of the vector store.
            bin_file (str): The path to the FAISS index file.
            id2img_fps (dict): A dictionary mapping IDs to image file paths.
            device (str): The device to use for computations (e.g., 'cuda' or 'cpu').
            embedding_model: The embedding model used for feature extraction.
        """
        self.root = root
        self.index = self.load_bin_file(bin_file)
        self.id2img_fps = id2img_fps
        self.device = device
        self.embedding_model = embedding_model

    def load_bin_file(self, bin_file: str) -> faiss.Index:
        """
        Loads the FAISS index from a binary file.

        Args:
            bin_file (str): The path to the FAISS index file.

        Returns:
            faiss.Index: The loaded FAISS index.
        """
        return faiss.read_index(bin_file)

    def _extract_features(self, image_path: str, online: bool = False) -> np.ndarray:
        """
        Extracts features from an image using the embedding model.

        Args:
            image_path (str): The path to the image file.
            online (bool): Whether to load the image online (default: False).

        Returns:
            np.ndarray: The extracted features.
        """
        if online:
            img = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        else:
            img = Image.open(image_path)
        
        if "blip" in self.embedding_model:
            image = self.embedding_model.image_processor["eval"](img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.embedding_model.model.extract_features({"image": image}, mode="image")
                return features.image_embeds_proj[:,0,:].detach().cpu().numpy()
        else:
            inputs = self.embedding_model.image_processor(images=img, return_tensors="pt").to(self.device)
            return self.embedding_model.model.get_image_features(**inputs).detach().cpu().numpy()

    def _extract_text_features(self, text: str) -> np.ndarray:
        """
        Extracts features from text using the embedding model.

        Args:
            text (str): The text to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        if "blip" in self.embedding_model:
            text_input = self.embedding_model.text_processor["eval"](text)
            sample = {"text_input": [text_input]}
            features = self.embedding_model.model.extract_features(sample, mode="text")
            return features.text_embeds_proj[:,0,:].detach().cpu().numpy()
        else:
            inputs = self.embedding_model.text_processor([text], return_tensors="pt").to(self.device)
            return self.embedding_model.model.get_text_features(**inputs).cpu().detach().numpy().astype(np.float32)

    def image_search(self, id_query: str, k: int = 10) -> tuple:
        """
        Searches for similar images to the given ID.

        Args:
            id_query (str): The ID to search for.
            k (int): The number of similar images to return (default: 10).

        Returns:
            tuple: A tuple containing the scores, indices, image paths, and info queries.
        """
        query_feats = self.index.reconstruct(id_query).reshape(1,-1)
        scores, idx_image = self.index.search(query_feats, k=k)
        idx_image = idx_image.flatten()
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info for info in infos_query]
        return scores, idx_image, infos_query, image_paths

    def text_search(self, text: str, k: int) -> tuple:
        """
        Searches for similar images to given text.

        Args:
            text (str): The text to search for.
            k (int): The number of similar images to return.

        Returns:
            tuple: A tuple containing the scores, indices, image paths, and info queries.
        """
        text_features = self._extract_text_features(text)
        scores, idx_image = self.index.search(text_features, k=k)
        idx_image = idx_image.flatten()
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info for info in infos_query]
        return scores, idx_image, infos_query, image_paths

    def image_similarity_search(self, image_path: str, k: int, online: bool = False) -> tuple:
        """
        Searches for similar images to a given image.

        Args:
            image_path (str): The path to the image file.
            k (int): The number of similar images to return.
            online (bool): Whether to load the image online (default: False).

        Returns:
            tuple: A tuple containing the scores, indices, image paths, and info queries.
        """
        image_features = self._extract_features(image_path, online)
        scores, idx_image = self.index.search(image_features, k=k)
        idx_image = idx_image.flatten()
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info for info in infos_query]
        return scores, idx_image, infos_query, image_paths
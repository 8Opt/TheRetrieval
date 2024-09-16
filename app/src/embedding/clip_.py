from app.src.base import BaseTool

from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer


class ClipTool(BaseTool): 
    
    def __init__(self, model_name:str="openai/clip-vit-base-patch32", device:str="auto") -> None:
        super().__init__()
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.text_processor = CLIPTokenizer.from_pretrained(model_name)

    def run(self, input):
        return super().run(input)
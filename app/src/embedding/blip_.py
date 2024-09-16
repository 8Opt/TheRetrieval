from app.src.base import BaseTool

from lavis.models import load_model_and_preprocess


class BlipTool(BaseTool): 

    def __init__(self, model_name:str="blip2_feature_extractor", 
                 model_type:str="pretrain", 
                 is_eval:bool=True,
                 device:str="auto") -> None:
        super().__init__()
        self.model_name = model_name
        self.model, self.image_processor, self.text_processor = load_model_and_preprocess(name=model_name,
                                                                        model_type=model_type,
                                                                        is_eval=is_eval,
                                                                        device=device)

    def run(self, input):
        return super().run(input)
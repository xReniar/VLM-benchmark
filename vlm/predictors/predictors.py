from abc import ABC, abstractmethod
from transformers import AutoProcessor
from PIL import Image
import torch

class HFPredictor(ABC):
    def __init__(self, model_name: str, args: dict):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.mps.is_available() else
            "cpu"
        )

        # predictor hyperparameters
        '''
        self.top_p = float(args["top_p"])
        self.top_k = float(args["top_k"])
        self.max_tokens = int(args["max_tokens"])
        '''

        # instantiate model
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self._init_model()

    def open_img(self, img_path: str) -> Image.Image:
        return Image.open(img_path)

    @abstractmethod
    def _init_model(self) -> None:
        pass
    
    @abstractmethod
    def inference(self, prompt: str, img_path: str):
        pass
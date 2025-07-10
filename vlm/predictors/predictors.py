from abc import ABC, abstractmethod
from transformers import AutoProcessor
import torch


class HFPredictor(ABC):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.mps.is_available() else
            "cpu"
        )

        # instantiate model
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self._init_model()

    def open_img(self, img_path: str):
        pass

    @abstractmethod
    def _init_model(self):
        pass
    
    @abstractmethod
    def inference(self, prompt):
        pass
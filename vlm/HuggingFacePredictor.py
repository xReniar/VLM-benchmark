from .predictor import Predictor
from abc import abstractmethod
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch


class HuggingFacePredictor(Predictor):
    def __init__(
        self,
        model_name: str
    ) -> None:
        super().__init__()
        self.model_name = model_name

    def _init_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText().from_pretrained(
            pretrained_model_name_or_path = self.model_name,
            torch_dtype = torch.bfloat16,
            _attn_implementation = "eager"
        ).to(self.device())

    @abstractmethod
    def inference(self):
        pass
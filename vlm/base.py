from abc import ABC, abstractmethod
from transformers import AutoProcessor
import torch
from PIL import Image


class VLMModelBase(ABC):
    _registry = {}

    def __init__(self, config: dict):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        self.model = self._init_model()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.mps.is_available() else
            "cpu"
        )

    def open_img(self, img_path: str) -> Image:
        return Image.open(img_path)

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def predict(self, img_path: str, prompt: str) -> str:
        pass

    @classmethod
    def register_model(cls, model_type: str):
        def decorator(subclass):
            if model_type in cls._registry:
                raise ValueError(f"'{model_type}' model type already registered")
            cls._registry[model_type] = subclass
            return subclass
        return decorator

    @classmethod
    def get_model_class(cls, model_type: str):
        return cls._registry.get(model_type)
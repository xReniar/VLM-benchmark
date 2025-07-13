from abc import ABC, abstractmethod
from transformers import AutoProcessor
import torch
from PIL import Image


class VLMModelBase(ABC):
    _registry = {}

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.mps.is_available() else
            "cpu"
        )
        
        # model base setup
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        self.params = self._init_params()
        self.model = self._init_model()

    def _init_params(self) -> dict:
        params = {}

        params["max_new_tokens"] = self.config.get("max_new_tokens", 3000)
        params["repetition_penalty"] = self.config.get("repetition_penalty")
        params["temperature"] = self.config.get("temperature")
        params["top_p"] = self.config.get("top_p")
        params["top_k"] = self.config.get("top_k")
        return params

    def open_img(self, img_path: str) -> Image:
        return Image.open(img_path)

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def predict(self, img_path: str, prompt: str) -> dict:
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
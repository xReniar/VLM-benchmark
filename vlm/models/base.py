from abc import ABC, abstractmethod
from transformers import AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
import torch


def get_torch_dtype(dtype: str) -> str:
    dtypes = {
        "int8": torch.int8,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto"
    }

    return dtypes[dtype]    

class VLMModelBase(ABC):
    _registry = {}

    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.mps.is_available() else
            "cpu"
        )
        
        # model base setup
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        self.params = self._init_params(self.config.get("parameters"))
        self.quantization = self._init_quantization(self.config.get("quantization"))
        self.torch_dtype = get_torch_dtype(self.config.get("torch_dtype", "auto"))
        self.attn_implementation = "eager"
        self.model = self._init_model()
        self.model.to(self.device)

    def _init_params(self, config_params: dict) -> dict:
        params = {}
        
        if config_params is not None:
            params["max_new_tokens"] = config_params.get("max_new_tokens", 1000)
            params["repetition_penalty"] = config_params.get("repetition_penalty")
            params["temperature"] = config_params.get("temperature")
            params["top_p"] = config_params.get("top_p")
            params["top_k"] = config_params.get("top_k")

            params = {k: v for k, v in params.items() if v is not None}

        params["do_sample"] = True if params else False
        
        return params
    
    def _init_quantization(self, config_quantization: dict) -> BitsAndBytesConfig | None:
        quantization = None
        
        # solve THIS
        if config_quantization is not None:
            quantization = {k: v for k, v in config_quantization.items() if v is not None}
            quantization = BitsAndBytesConfig(**quantization)
        
        return quantization

    def open_img(self, img_path: str) -> Image.Image:
        return Image.open(img_path)

    @abstractmethod
    def _init_model(self) -> object:
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
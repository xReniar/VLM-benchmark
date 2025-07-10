from abc import ABC, abstractmethod
import torch
import json


class Predictor(ABC):
    def __init__(
        self
    ) -> None:
        super().__init__()

        # init model and other stuff
        self._init_model()

    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def inference(self, prompt: str, img_path: str):
        pass
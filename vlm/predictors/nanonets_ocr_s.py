from .predictors import HFPredictor
from transformers import AutoModelForVision2Seq
import torch


class QwenVL(HFPredictor):
    def __init__(self, model_name):
        super().__init__(model_name)

    def _init_model(self):
        self.model = None

    def inference(self, prompt: str, img_path: str):
        pass
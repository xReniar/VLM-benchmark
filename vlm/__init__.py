from .base import VLMModelBase
from . import gemma3n
from . import imagetext2text
from . import qwen2_5_vl
from . import vision2seq
import yaml


class VLM():
    def __init__(
        self,
        model_name: str
    ) -> None:
        self.config = yaml.safe_load(open("./configs/models.yaml", "r"))[model_name]
        self._init_model()

    def _init_model(self):
        model_type = self.config["type"]
        ModelClass = VLMModelBase.get_model_class(model_type)

        self.model: VLMModelBase = ModelClass(self.config)

    def predict(
        self,
        img_path: str,
        prompt: str
    ) -> dict[str, float]:
        return self.model.predict(img_path, prompt)
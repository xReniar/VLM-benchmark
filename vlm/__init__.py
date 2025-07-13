from .base import VLMModelBase
import yaml


class VLM():
    def __init__(
        self,
        model_name: str
    ) -> None:
        self.config = yaml.safe_load(open("../configs/models.yaml", "r"))[model_name]
        self._init_model()

    def _init_model(self):
        model_type = self.config["type"]
        ModelClass = VLMModelBase.get_model_class(model_type)

        self.model = ModelClass(self.config)

    def predict(
        self,
        img_path: str, 
        prompt: str
    ) -> str:
        return self.model.predict(img_path, prompt)
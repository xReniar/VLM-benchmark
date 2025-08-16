from .models import VLMModelBase
import yaml


class VLM():
    def __init__(
        self,
        model_name: str
    ) -> None:
        config = yaml.safe_load(open("./configs/models.yaml", "r"))[model_name]
        ModelClass = VLMModelBase.get_model_class(config["type"])

        self.model: VLMModelBase = ModelClass(config)

    def predict(
        self,
        img_path: str,
        prompt: str
    ) -> dict[str, float]:
        return self.model.predict(img_path, prompt)
    
__all__ = [
    "VLM"
]
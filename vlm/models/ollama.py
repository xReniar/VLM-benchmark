from .base import VLMModelBase
import base64
import requests
import time


@VLMModelBase.register_model("Ollama")
class OllamaVLM(VLMModelBase):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def _init_model(self) -> object:
        pass

    def predict(self, img_path: str, prompt: str) -> dict:
        with open(img_path, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        payload = dict(
            model = self.config["model_id"],
            prompt = prompt,
            images = image_base64,
            stream = False,
            options = self.params
        )

        start = time.time()
        response = requests.post(
            url = "http://localhost:1134/api/generate",
            json = payload
        )
        end = time.time()

        result = dict(t = end - start)
        
        if response.status_code == 200:
            result["response"] = response.json()
        else:
            result["response"] = dict(
                staus_code = response.status_code,
                error = response.text,
                image = img_path,
            )

        return result

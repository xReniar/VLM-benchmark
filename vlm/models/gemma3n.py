from .base import VLMModelBase
from transformers import Gemma3ForConditionalGeneration
import base64
import torch
import time


@VLMModelBase.register_model("Gemma3n")
class Gemma3n(VLMModelBase):
    def __init__(self, config: dict):
        super().__init__(config)

    def _init_model(self):
        return Gemma3ForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            torch_dtype="auto",
            device_map="auto"
        ).eval()
    
    def predict(self, img_path: str, prompt: str) -> dict:
        with open(img_path, "rb") as img_file:
            img =  base64.b64encode(img_file.read()).decode('utf-8')

        messages = [{
            "role": "user",
            "content": [
                { "type": "image", "image": img },
                { "type": "text", "text": prompt }
            ]
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        start = time.time()
        with torch.inference_mode():
            generation = self.model.generate(**inputs, **self.params, do_sample=False)
            generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        end = time.time()

        return dict(
            response = decoded,
            t = end - start
        )
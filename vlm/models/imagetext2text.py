from .base import VLMModelBase
from transformers import AutoModelForImageTextToText
import torch
import time


@VLMModelBase.register_model("ImageTextToText")
class ImageTextToText(VLMModelBase):
    def __init__(self, config: dict):
        super().__init__(config)

    def _init_model(self):
        return AutoModelForImageTextToText.from_pretrained(
            self.config["model_id"],
            torch_dtype=self.torch_dtype,
            _attn_implementation=self.attn_implementation,
            quantization_config = self.quantization
        ).to(self.device)

    def predict(self, img_path: str, prompt: str) -> dict:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "path": img_path},
                {"type": "text", "text": prompt}
            ]
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        start = time.time()
        generated_ids = self.model.generate(**inputs, **self.params)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        end = time.time()
        
        return dict(
            response = generated_texts[0],
            t = end - start
        )
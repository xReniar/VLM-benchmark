from .base import VLMModelBase
from transformers import AutoModelForImageTextToText
import torch


@VLMModelBase.register_model("ImageText2Text")
class ImageText2Text(VLMModelBase):
    def __init__(self, config: dict):
        super().__init__(config)

    def _init_model(self):
        return AutoModelForImageTextToText.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager"
        ).to(self.device)

    def predict(self, img_path: str, prompt: str) -> str:
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

        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        return generated_texts[0]
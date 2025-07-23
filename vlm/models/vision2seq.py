from .base import VLMModelBase
from transformers import AutoModelForVision2Seq
import torch
import time


@VLMModelBase.register_model("Vision2Seq")
class Vision2Seq(VLMModelBase):
    def __init__(self, config: dict):
        super().__init__(config)

    def _init_model(self):
        return AutoModelForVision2Seq.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",
            quantization_config = self.quantization
        ).to(self.device)

    def predict(self, img_path: str, prompt: str) -> dict:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]

        img = self.open_img(img_path)

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            images=img,
            return_tensors="pt"
        ).to(self.device)

        start = time.time()
        generated_ids = self.model.generate(**inputs, **self.params, do_sample=True)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        end = time.time()
        img.close()

        return dict(
            response = generated_texts[0],
            t = end - start
        )
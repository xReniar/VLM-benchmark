from .predictors import HFPredictor
from transformers import AutoModelForVision2Seq
import torch


class SmolVLM(HFPredictor):
    def __init__(self, model_name):
        super().__init__(model_name)

    def _init_model(self):
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16
        ).to(self.device)

    def inference(self, prompt, img_path):
        messages = [{
            "role": "user",
            "content": [ {"type": "image"}, {"type": "text", "text": prompt} ]
        }]

        img = self.open_img(img_path)

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[img], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.sprocessor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_texts[0]
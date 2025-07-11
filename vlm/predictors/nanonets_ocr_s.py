from .predictors import HFPredictor
from transformers import AutoModelForImageTextToText
import torch


class Nanonets(HFPredictor):
    def __init__(self, model_name):
        super().__init__(model_name)

    def _init_model(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, 
            torch_dtype="auto"
        ).to(self.device)

    def inference(self, prompt: str, img_path: str):
        img = self.open_img(img_path)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": prompt}
            ]}
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[img], padding=True, return_tensors="pt").to(self.device)
        
        output_ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]
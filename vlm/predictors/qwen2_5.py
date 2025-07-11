from .predictors import HFPredictor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText, TorchAoConfig
from qwen_vl_utils import process_vision_info
import torch


# "Qwen/Qwen2.5-VL-3B-Instruct"
class Qwen2_5_VL(HFPredictor):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)

    def _init_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16
        ).to(self.device)

    def inference(self, prompt: str, img_path: str):
        messages = [{
            "role": "user",
            "content": [
                { "type": "image", "image": img_path },
                { "type": "text", "text": prompt }
            ]
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text

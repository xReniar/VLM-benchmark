from .base import VLMModelBase
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import base64
import time


@VLMModelBase.register_model("Qwen2.5-VL")
class Qwen2_5_VL(VLMModelBase):
    def __init__(self, config: dict):
        super().__init__(config)

    def _init_model(self):
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            torch_dtype="auto",
            device_map="auto"
        )
    
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

        # Preparation for inference
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
        )
        inputs = inputs.to("cuda")

        start = time.time()
        generated_ids = self.model.generate(**inputs, **self.params, do_sample=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        end = time.time()
        
        return dict(
            response = output_text,
            t = end - start
        )
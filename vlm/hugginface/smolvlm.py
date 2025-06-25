from ..HuggingFacePredictor import HuggingFacePredictor
from PIL import Image
import torch


# model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
class SmolVLM(HuggingFacePredictor):
    def __init__(
        self,
        model_name: str
    ) -> None:
        super().__init__(model_name)

    def inference(
        self,
        img_path: str,
        prompt: str
    ):
        img = Image.open(img_path)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.3,
            max_new_tokens=1000
        )

        return self.processor.decode(outputs[0], skip_special_tokens=True)
#from ..HuggingFacePredictor import HuggingFacePredictor


import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForVision2Seq
from PIL import Image


model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id,
                                                torch_dtype=torch.bfloat16,
                                                _attn_implementation="eager").to(DEVICE)


def run_inference(image: Image.Image, prompt: str):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]

    #inputs = processor.apply_chat_template(messages, add_generation_prompt=True)
    # Step 2: Process image + prompt text into model inputs
    #model_inputs = processor(text=inputs, images=[image], return_tensors="pt").to(DEVICE)

    #outputs = model.generate(**model_inputs)


    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    outputs = model.generate(**inputs, do_sample=True, temperature=0.1, top_p=0.9, top_k=0, repetition_penalty=1.3, max_new_tokens=1000)

    return processor.decode(outputs[0], skip_special_tokens=True)

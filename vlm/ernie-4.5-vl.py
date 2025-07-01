import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from PIL import Image



model_path = 'baidu/ERNIE-4.5-VL-28B-A3B-PT'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
processor.eval()
model.add_image_preprocess(processor)


image = Image.open(f"../datasets/kie/images/0.png")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image."},
            {"type": "image", "image": f"{image}"},
        ]
    },
]


text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
image_inputs, video_inputs = processor.process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

device = next(model.parameters()).device
inputs = inputs.to(device)

generated_ids = model.generate(
    inputs=inputs['input_ids'].to(device),
    **inputs,
    max_new_tokens=128
    )
output_text = processor.decode(generated_ids[0])
print(output_text)
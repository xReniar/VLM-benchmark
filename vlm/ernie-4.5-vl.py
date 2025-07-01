import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import os
import json


def inference(
    image: Image.Image,
    model,
    processor,
    prompt: str
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
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

    return output_text


model_path = 'baidu/ERNIE-4.5-VL-28B-A3B-PT'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    load_in_4_bit=True
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
processor.eval()
model.add_image_preprocess(processor)


images = sorted(os.listdir("../datasets/kie/images"))
labels = sorted(os.listdir("../datasets/kie/labels"))


os.makedirs("../responses/kie/ernie-4.5", exist_ok=True)

field_names = ["date", "doc_no_receipt_no", "seller_address", "seller_gst_id", "seller_name", "seller_phone", "total_amount", "total_tax"]
output_format = {field: ".." for field in field_names}

prompt = "Extract the following {fields} from the above document. If a field is not present, return ''. Return the output in a valid JSON format like {output_format}" \
    .format(
        fields = field_names,
        output_format = output_format
    )

for (img_fn, _) in zip(images, labels):
    if not os.path.exists(f"../responses/kie/ernie-4.5/{img_fn}.json"):
        image = Image.open(f"../datasets/kie/images/{img_fn}")
        inference_time, result = inference(
            image = image,
            model = model,
            processor = processor,
            prompt = prompt
        )

        '''json_str = result.split("Assistant: ")[1]
        try:
            json_result = json.loads(json_str)
        except:
            json_result = {}'''
        json_result = dict(
            prediction = result,
            inference_time = inference_time
        )


        with open(f"../responses/kie/ernie-4.5/{img_fn}.json", "w") as f:
            json.dump(json_result, f, indent=4)
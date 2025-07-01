import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForVision2Seq
from PIL import Image
import os
import json
import time


def inference(
    image: Image.Image,
    model,
    processor,
    prompt: str
):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    start = time.time()
    generated_ids = model.generate(
        **inputs,
        do_sample=True, 
        #temperature=0.1, 
        top_p=0.9, 
        #top_k=0, 
        repetition_penalty=1.3, 
        max_new_tokens=1000
    )
    end = time.time()

    #return processor.decode(outputs[0], skip_special_tokens=True)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    return (end - start, generated_texts[0])


images = sorted(os.listdir("../datasets/kie/images"))
labels = sorted(os.listdir("../datasets/kie/labels"))


os.makedirs("../responses/kie/smolvlm", exist_ok=True)

field_names = ["date", "doc_no_receipt_no", "seller_address", "seller_gst_id", "seller_name", "seller_phone", "total_amount", "total_tax"]
output_format = {field: ".." for field in field_names}

prompt = "Extract the following {fields} from the above document. If a field is not present, return ''. Return the output in a valid JSON format like {output_format}" \
    .format(
        fields = field_names,
        output_format = output_format
    )

#model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager"
).to(DEVICE)

for (img_fn, _) in zip(images, labels):
    if not os.path.exists(f"../responses/kie/smolvlm/{img_fn}.json"):
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


        with open(f"../responses/kie/smolvlm/{img_fn}.json", "w") as f:
            json.dump(json_result, f, indent=4)
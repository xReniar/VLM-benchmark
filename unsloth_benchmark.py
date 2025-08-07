from unsloth import FastVisionModel
from PIL import Image
import os
import time
import json

system_message = """You are a highly advanced Vision Language Model (VLM), specialized in extracting visual data. 
Your task is to process and extract meaningful insights from images, leveraging multimodal understanding
to provide accurate and contextually relevant information."""


model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "training/single_test/qwen2.5-vl_sroie/lora_model",
    load_in_4bit = True
)

FastVisionModel.for_inference(model)

imgs_folder_path = "data/sroie/test/img"
imgs_folder = sorted(os.listdir(imgs_folder_path))

field_names = ['address', 'company', 'date', 'total']
output_format = {field: ".." for field in field_names}


prompt = "Extract the following {fields} from the above document. If a field is not present, return ''. Return the output in a valid JSON format like {output_format}" \
    .format(
        fields = field_names,
        output_format = output_format
    )

result = {}

for fn in imgs_folder:
    img = Image.open(f"{imgs_folder_path}/{fn}")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                { "type": "image" },
                { "type": "text", "text": prompt }
            ]
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        img,
        input_text,
        add_special_tokens = False,
        truncation=False,
        return_tensors = "pt",
    ).to("cuda")

    start = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        use_cache=True,
        temperature=1.5,
        min_p=0.1
    )
    end = time.time()

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    result[fn] = dict(
        response = output_text,
        t = end - start
    )

    with open("prova.json", "w") as f:
        json.dump(result, f, indent=4)
import os
import sys
sys.path.append(f"{os.path.dirname(os.getcwd())}")

from dataset import *
from PIL import Image
import json
import base64
from openai import OpenAI
import time

client = OpenAI(base_url=f"http://localhost:8080/v1", api_key="token-abc123")

system_message = """You are a highly advanced Vision Language Model (VLM), specialized in extracting visual data.
Your task is to process and extract meaningful insights from images that are asked in the prompt."""

imgs_fn = []

def format_data(sample):
    with open(sample.image_path, "rb") as img_file:
        pil_image = base64.b64encode(img_file.read()).decode("utf-8")

    field_names = set([entity.label for entity in sample.entities])

    output_format = {field: ".." for field in field_names}

    prompt = "Extract the following {fields} from the above document. If a field is not present, return ''. Return the output in a valid JSON format like {output_format}" \
        .format(
            fields = list(field_names),
            output_format = output_format
        )
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{pil_image}"
                    }
                },
                { "type": "text", "text": prompt }
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": json.dumps(sample.to_json("kie"))
            }]
        }
    ]

    return dict(
        conversation = conversation,
        img_path = sample.image_path
    )


dataset = "docile"

if dataset == "docile":
    test_dataset = [format_data(sample) for sample in DocILE(tasks=["kie"], split="val")]
else:
    test_dataset = [format_data(sample) for sample in SROIE(tasks=["kie"], split="test")]

print(len(test_dataset))


result = {}

for obj in test_dataset:
    img_fn = obj["img_path"].split("/")[-1]

    try:
        start = time.time()
        response = client.chat.completions.create(
            model=f"xReniar/Gemma-3-4B-{dataset}_ft",
            messages = obj["conversation"][0:2],
            max_tokens = 3000,
            #extra_body={"repetition_penalty": 1}
        )
        end = time.time()

        inference_time = end - start

        response = response.choices[0].message.content
        response = response.strip(" ```json\n").strip("\n```")
    except:
        response = {}
        inference_time = -1

    result[img_fn] = {
        "response": response,
        "time": inference_time
    }

    with open(f"gemma-3-4B-{dataset}.json", "w") as f:
        json.dump(result, f, indent=4)
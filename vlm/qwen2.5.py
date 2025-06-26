import base64
import requests
import os
import json


def inference(image_path, prompt, options, model='qwen2.5vl:3b'):
    # Carica e codifica l'immagine in base64
    with open(image_path, 'rb') as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    # Prepara la richiesta al modello Ollama
    payload = {
        'model': model,
        'prompt': prompt,
        'images': [image_base64],
        'stream': False,
        'options': options or {}

    }

    response = requests.post('http://localhost:11434/api/generate', json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Errore: {response.status_code} - {response.text} - {image_path}")
    

images = sorted(os.listdir("../datasets/kie/images"))
labels = sorted(os.listdir("../datasets/kie/labels"))
os.makedirs("../responses/kie/qwen2.5", exist_ok=True)

field_names = ["date", "doc_no_receipt_no", "seller_address", "seller_gst_id", "seller_name", "seller_phone", "total_amount", "total_tax"]
output_format = {field: ".." for field in field_names}


prompt = "Extract the following {fields} from the above document. If a field is not present, return ''. Return the output in a valid JSON format like {output_format}" \
    .format(
        fields = field_names,
        output_format = output_format
    )

options = {
  'num_predict': 1024,
}

for (img_fn, _) in zip(images, labels):
    if not os.path.exists(f"../responses/kie/qwen2.5/{img_fn}.json"):
        #image = Image.open(f"../datasets/kie/images/{img_fn}")
        img_path = f"../datasets/kie/images/{img_fn}"
        response = inference(
            image_path=img_path,
            prompt=prompt,
            options=options
        )

        #response_str = response["response"]
        #s_clean = response_str.replace('```json', '').replace('```', '').strip()

        with open(f"../responses/kie/qwen2.5/{img_fn}.json", "w") as f:
            json.dump(response, f, indent=4)


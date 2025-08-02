import os
import json
import ast


def parse_smolvlm(file_name: str):
    folder_path = "responses/raw"
    predictions: dict = json.load(open(f"{folder_path}/{file_name}.json", "r"))
    os.makedirs("responses/parsed", exist_ok=True)

    parsed_predictions = {}

    for img_fn in predictions.keys():
        prediction = predictions[img_fn]

        response: str = prediction["response"]
        inference_time: float = prediction["t"]

        try:
            pred = response.split("Assistant: ")[1]
            pred = ast.literal_eval(pred)
            pred = json.loads(json.dumps(pred))

            if not isinstance(pred, dict):
                raise ValueError("Parsed prediction is not a dictionary")
        except:
            pred = {}
        finally:
            parsed_predictions[img_fn] = dict(
                response = pred,
                inference_time = inference_time
            )

    with open(f"responses/parsed/{file_name}.json", "w") as f:
        json.dump(parsed_predictions, f, indent=4)


def parse_qwen2_5(file_name: str):
    folder_path = "responses/raw"
    predictions: dict = json.load(open(f"{folder_path}/{file_name}.json", "r"))
    os.makedirs("responses/parsed", exist_ok=True)

    parsed_predictions = {}

    for img_fn in predictions.keys():
        prediction = predictions[img_fn]

        response: str = prediction["response"]
        inference_time: float = prediction["inference_time"]

        try:
            pred = response.replace('```json', '').replace('```', '').strip()
            pred = json.loads(pred)

            if not isinstance(pred, dict):
                raise ValueError("Parsed prediction is not a dictionary")
        except:
            pred = {}
        finally:
            parsed_predictions[img_fn] = dict(
                response = pred,
                inference_time = inference_time
            )

    with open(f"responses/parsed/{file_name}.json", "w") as f:
        json.dump(parsed_predictions, f, indent=4)



#parse_smolvlm("SmolVLM-500M-kie")
#parse_smolvlm("SmolVLM-kie")
#parse_smolvlm("SmolVLM2-kie")
parse_qwen2_5("qwen2.5-7b")
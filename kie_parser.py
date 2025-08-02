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


def parse_qwen2_5():
    folder_path = "responses/raw/kie/qwen2.5"
    responses = sorted(os.listdir(folder_path))

    for response in responses:
        json_response = json.load(open(os.path.join(folder_path, response), "r"))

        response_field: str = json_response["response"]

        prediction_json = {}
        try:
            prediction = response_field.replace('```json', '').replace('```', '').strip()
            prediction_json = json.loads(prediction)
        except:
            prediction_json = {}

        with open(os.path.join(folder_path.replace("raw", "parsed"), response), "w") as f:
            json.dump(prediction_json, f, indent=4)



parse_smolvlm("SmolVLM-500M-kie")
parse_smolvlm("SmolVLM-kie")
parse_smolvlm("SmolVLM2-kie")
#parse_qwen2_5()
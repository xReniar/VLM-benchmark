import os
import json
import ast


def parse_smolvlm():
    '''
    folder_path = "responses/raw/kie/smolvlm"
    responses = sorted(os.listdir(folder_path))

    for response in responses:
        json_response = json.load(open(os.path.join(folder_path, response), "r"))

        response_field:str = json_response["prediction"]

        prediction_json = {}
        try:
            prediction = response_field.split("Assistant: ")[1]
            prediction_json = json.loads(prediction)
        except:
            pass
    
        with open(os.path.join(folder_path.replace("raw", "parsed"), response), "w") as f:
            json.dump(prediction_json, f, indent=4)
    '''
    folder_path = "responses/raw"
    file_name = "SmolVLM-500M-kie.json"
    predictions: dict = json.load(open(f"{folder_path}/{file_name}", "r"))
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


            parsed_predictions[img_fn] = dict(
                response = pred,
                inference_time = inference_time
            )
        except:
            parsed_predictions[img_fn] = dict(
                response = {},
                inference_time = inference_time
            )

    with open(f"responses/parsed/{file_name}", "w") as f:
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



parse_smolvlm()
#parse_qwen2_5()
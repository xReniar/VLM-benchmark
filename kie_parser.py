import os
import json


def parse_smolvlm():
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
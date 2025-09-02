import os
import json
import re


def parse(dataset: str, vlm: str):
    folder_name = f"result-{dataset}"

    parsed = {}
    if vlm == "qwen":
        json_name = "Qwen2.5-VL-normal-r8.json"
        preds = json.load(open(f"{folder_name}/raw/{json_name}"))
        for key in preds.keys():
            obj = preds[key]
            
            try:
                response = json.loads(re.search(r"```json\s*(.*?)\s*```", obj["response"].split("assistant")[1], re.DOTALL).group(1).strip())
            except:
                response = {}

            parsed[key] = dict(
                response = response,
                inference_time = obj["t"]
            )

        with open(f"{folder_name}/parsed/{json_name}", "w") as f:
            json.dump(parsed, f, indent=4)

    if vlm == "smol":
        json_name = "smolvlm2-normal-r8.json"
        preds = json.load(open(f"{folder_name}/raw/{json_name}"))
        for key in preds.keys():
            obj = preds[key]
            try:
                #response = json.loads(obj["response"].split("Assistant: ")[1])
                #print(response)
                response = json.loads(re.search(r"```json\n\s*(.*?)\s*\n```", obj["response"].split("Assistant: ")[1], re.DOTALL).group(1).strip())
            except:
                response = {}
            
            parsed[key] = dict(
                response = response,
                inference_time = obj["inference_time"]
            )

        with open(f"{folder_name}/parsed/{json_name}", "w") as f:
            json.dump(parsed, f, indent=4)


parse(
    dataset="docile",
    vlm="smol"
)

import os
import json


def metric(dataset: str):
    folder_path = f"result-{dataset}/parsed"
    files = sorted(os.listdir(folder_path))

    for file in files:
        file_dict = json.load(open(f"{folder_path}/{file}", mode="r"))
        
        mean_time = 0
        for key in file_dict.keys():
            obj = file_dict[key]

            inference_time = obj["inference_time"]
            if inference_time < 0:
                inference_time = 1

            mean_time += inference_time

        print(file, mean_time / len(file_dict.keys()))




metric("docile")
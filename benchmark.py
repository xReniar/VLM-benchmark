from vlm import VLM
import benchmark
import torch
import argparse
import os
import json
import yaml
import gc


def inference(
    task: str,
    dataset: str,
    models: list[str]
):
    prompt = "extract "
    folder_path = "data/kie/images"
    data_folder = sorted(os.listdir(folder_path))

    os.makedirs(f"responses/raw", exist_ok=True)
    
    for model_name in models:
        output_dict = {}
        model = VLM(model_name)

        for fn in data_folder:
            output_dict[fn] = model.predict(f"{folder_path}/{fn}", prompt)

        with open(f"responses/raw/{model_name}-{task}.json") as f:
            json.dump(output_dict, f, indent=4)

        # clear memory
        torch.cuda.empty_cache()
        del model
        gc.collect()


if __name__ == "__main__":
    benchmark_config = yaml.safe_load(open("./configs/benchmark.yaml", "r"))
    test_config = benchmark_config["test"]
    tasks_config = benchmark_config["tasks"]

    TASK: str = test_config["task"]
    DATASET: str = test_config["dataset"]
    MODELS: list[str] = test_config["models"]

    task_config: dict = tasks_config[TASK]

    inference(TASK, DATASET, MODELS)
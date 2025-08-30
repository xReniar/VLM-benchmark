from vlm import VLM
from dataset import Dataset, Data, SROIE, DocILE, Task
import benchmark
import torch
import argparse
import os
import json
import yaml
import gc


def generate_prompt(task: str, data: Data) -> str:
    prompt = ""
    if task == Task.KIE.value:
        kie_data = data.to_json(task)
        field_names = list(kie_data.keys())
        output_format = {field: ".." for field in field_names}

        prompt = "Extract the following {fields} from the above document. If a field is not present, return ''. Return the output in a valid JSON format like {output_format}" \
            .format(
                fields = field_names,
                output_format = output_format
            )

    if task == Task.OCR.value:
        print(task)
    if task == Task.VQA.value:
        print(task)
    return prompt


def inference(
    task: str,
    dataset: str,
    models: list[str]
):
    output_dir = f"responses/raw/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in models:
        output_dict = {}
        model = VLM(model_name)

        for data in SROIE(tasks=[task], split="test"):
            fn = data.image_path.split("/")[-1]

            output_dict[fn] = model.predict(
                img_path=data.image_path,
                prompt=generate_prompt(task, data)
            )

            with open(f"{output_dir}/{model_name}-{task}.json", "w") as f:
                json.dump(output_dict, f, indent=4)

        # clear memory
        torch.cuda.empty_cache()
        del model
        gc.collect()


if __name__ == "__main__":
    benchmark_config = yaml.safe_load(open("./configs/benchmark.yaml", "r"))
    test_config: dict = benchmark_config["test"]

    TASK: str = test_config["task"]
    DATASET: str = test_config["dataset"]
    MODELS: list[str] = test_config["models"]

    inference(TASK, DATASET, MODELS)
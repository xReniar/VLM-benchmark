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
    folder_path = f"data/{dataset}/test/img"
    output_dir = f"responses/raw/{dataset}"
    os.makedirs(output_dir, exist_ok=True)

    #field_names = ["date", "doc_no_receipt_no", "seller_address", "seller_gst_id", "seller_name", "seller_phone", "total_amount", "total_tax"]
    field_names = ['address', 'company', 'date', 'total']
    output_format = {field: ".." for field in field_names}


    prompt = "Extract the following {fields} from the above document. If a field is not present, return ''. Return the output in a valid JSON format like {output_format}" \
        .format(
            fields = field_names,
            output_format = output_format
        )
    
    for model_name in models:
        output_dict = {}
        model = VLM(model_name)

        for fn in sorted(os.listdir(folder_path)):
            output_dict[fn] = model.predict(
                img_path=f"{folder_path}/{fn}",
                prompt=prompt
            )

            with open(f"{output_dir}/{model_name}-{task}.json", "w") as f:
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
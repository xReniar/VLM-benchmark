from datasets import load_dataset
from PIL import Image
import argparse
import io
import os
import json
import inspect


def kie():
    output_path = os.path.join("datasets", "kie")
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)

    for i, row in enumerate(load_dataset("nanonets/key_information_extraction", split="test")):
        image_bytes = row["image"]
        annotations = row["annotations"]

        with open(os.path.join(output_path, "labels", f"{i}.json"), "w") as file:
            json.dump(annotations, file, indent=4)

        image = Image.open(io.BytesIO(image_bytes))
        image.save(os.path.join(output_path, "images", f"{i}.png"))
        image.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset downloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--kie", action="store_true", help="Download dataset for KIE")
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    task_func = {
        "kie": kie
    }

    args_dict = args.__dict__
    for task in args_dict.keys():
        if args_dict[task]:
            task_func[task]()

    '''
    functions = []
    for name, obj in list(globals().items()):
        print(name, obj)
    '''
from datasets import load_dataset
from PIL import Image
import argparse
import io
import os
import json


'''
ds = load_dataset("nanonets/key_information_extraction", split="test")

for x in ds:
    print(x.keys())
'''

def kie():
    output_path = os.path.join("data", "kie")

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        for i, row in enumerate(load_dataset("nanonets/key_information_extraction", split="test")):
            image_bytes = row["image"]
            annotations = row["annotations"]

            with open(os.path.join(output_path, f"{i}.json"), "w") as file:
                json.dump(annotations, file, indent=4)

            image = Image.open(io.BytesIO(image_bytes))
            image.save(os.path.join(output_path, f"{i}.png"))
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

    if args.kie:
        kie()
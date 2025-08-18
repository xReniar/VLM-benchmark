from .ds import Dataset, Data, Task
from typing import ClassVar
from datasets import load_dataset
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os


class NNTS_KIE(Dataset):
    TASKS: ClassVar[list[Task]] = [Task.KIE]
    dataset_name: str = "nnts_kie"

    def __init__(
        self,
        tasks: list[Task],
        split: str
    ) -> None:
        super().__init__(tasks=tasks, split=split)

    def _download(self) -> None:
        output_path = f"{self.CACHE_DIR}/{self.dataset_name}/test"
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "images"))
        os.makedirs(os.path.join(output_path, "labels"))

        ds = load_dataset("nanonets/key_information_extraction", split="test")

        def process_item(i, row):
            json_path = os.path.join(output_path, "labels", f"{i}.json")
            with open(json_path, "w") as file:
                json.dump(row["annotations"], file, indent=4)

            image_path = os.path.join(output_path, "images", f"{i}.png")
            with Image.open(io.BytesIO(row["image"])) as img:
                img.save(image_path)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            executor.map(lambda x: process_item(x[0], x[1]), enumerate(ds))


    def _load_data(self) -> None:
        dir_path = f"{self.CACHE_DIR}/{self.dataset_name}"
        images = self.read_folder(f"{dir_path}/{self.split}/images")

        for image in images:
            label_name = image.replace(".png", ".json")
            entities = []

            if Task.KIE in self.tasks:
                json_f: dict = json.load(open(f"{dir_path}/{self.split}/labels/{label_name}", "r"))

                for key, item in json_f.items():
                    entities.append(self._convert_to_format(
                        task = Task.KIE,
                        item = dict(
                            label = key,
                            value = item
                        )
                    ))

            self.data.append(Data(
                image_path=f"{dir_path}/{self.split}/images/{image}",
                entities=entities
            ))
from .ds import Dataset, Data, Task
from typing import ClassVar
import json
import os
import requests
import zipfile
import shutil


class SROIE(Dataset):
    TASKS: ClassVar[list[Task]] = [Task.OCR, Task.KIE]
    dataset_name: str = "sroie"

    def __init__(
        self,
        tasks: list[Task],
        split: str
    ) -> None:
        super().__init__(tasks=tasks, split=split)

    def __extract_bbox_and_text(self, line: str) -> tuple[tuple[int], str]:
        coords = [int(x) for x in line[:8]]
        text = line[8] if len(line) > 8 else ""

        coords = [coords[0], coords[1], coords[0], coords[1]]
        for i in range(0, len(coords), 2):
            coords[0] = min(coords[0], coords[i])
            coords[1] = min(coords[1], coords[i + 1])
            coords[2] = max(coords[2], coords[i])
            coords[3] = max(coords[3], coords[i + 1])
        
        return coords, text
    
    def _download(self) -> None:
        url = "https://www.kaggle.com/api/v1/datasets/download/urbikn/sroie-datasetv2"
        response = requests.get(url)
        dir_path = f"{self.CACHE_DIR}/{self.dataset_name}"
        os.makedirs(dir_path)

        with open(f"{dir_path}/sroie.zip", "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(f"{dir_path}/sroie.zip", 'r') as zip_ref:
            zip_ref.extractall(dir_path)

        os.remove(f"{dir_path}/sroie.zip")
        for split in ["train", "test"]:
            shutil.move(
                src = f"{dir_path}/SROIE2019/{split}",
                dst = f"{dir_path}"
            )
        shutil.rmtree(f"{dir_path}/SROIE2019")
        


    def _load_data(self) -> None:
        dir_path = f"{self.CACHE_DIR}/{self.dataset_name}"
        images = self.read_folder(f"{dir_path}/{self.split}/img")

        for image in images:
            label = image.replace(".jpg", ".txt")
            fields, entities = [], []

            # For OCR task
            if Task.OCR in self.tasks:
                with open(f"{dir_path}/{self.split}/box/{label}", "r") as f:
                    rows = f.readlines()
                    rows.sort()
                    for row in rows:
                        row = row.strip("\n").split(',', 8)
                        if len(row) >= 8:
                            coords, text = self.__extract_bbox_and_text(row)

                            fields.append(self._convert_to_format(
                                task=Task.OCR,
                                item = dict(
                                    bbox = coords,
                                    text = text
                                )
                            ))

            # For KIE task
            if Task.KIE in self.tasks:
                with open(f"{dir_path}/{self.split}/entities/{label}", "r") as f:
                    json_f: dict = json.load(f)
                    for key, item in json_f.items():
                        entities.append(self._convert_to_format(
                            task=Task.KIE,
                            item = dict(
                                label = key,
                                value = item
                            )
                        ))

            self.data.append(Data(
                image_path=f"{dir_path}/{self.split}/img/{image}",
                fields=fields,
                entities=entities
            ))
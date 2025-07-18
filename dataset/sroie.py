from .ds import Dataset, Data, Task
import json
import os


class SROIE(Dataset):
    def __init__(
        self,
        task: Task,
        split: str
    ) -> None:
        super().__init__(task=task, split=split)
        self._load_data()

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

    def _load_data(self) -> None:
        images = self.read_folder(f"./data/sroie/{self.split}/img")

        for image in images:
            label = image.replace(".jpg", ".txt")

            # OCR task
            if self.task == Task.OCR:
                fields = []
                with open(f"./data/sroie/{self.split}/box/{label}", "r") as f:
                    rows = f.readlines()
                    rows.sort()
                    for row in rows:
                        row = row.strip("\n").split(',', 8)
                        if len(row) >= 8:
                            coords, text = self.__extract_bbox_and_text(row)

                            fields.append(self._convert_to_format(
                                task = self.task,
                                item = dict(
                                    bbox = coords,
                                    text = text
                                )
                            ))

                self.data.append(Data(
                    image_path=f"./data/sroie/{self.split}/img/{image}",
                    fields = fields
                ))

            # KIE task
            if self.task == Task.KIE:
                entities = []
                with open(f"./data/sroie/{self.split}/entities/{label}", "r") as f:
                    json_f: dict = json.load(f)
                    for key, item in json_f.items():
                        entities.append(self._convert_to_format(
                            task = self.task,
                            item = dict(
                                label = key,
                                value = item
                            )
                        ))

                self.data.append(Data(
                    image_path=f"./data/sroie/{self.split}/img/{image}",
                    entities=entities
                ))
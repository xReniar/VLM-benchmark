from .ds import Dataset, Data, Task
from typing import ClassVar
from pdf2image import convert_from_path
from PIL import Image
import json
import os


class DocILE(Dataset):
    TASKS: ClassVar[list[Task]] = [Task.OCR, Task.KIE]
    dataset_name: str = "docile"

    def __init__(
        self,
        tasks: list[Task],
        split: str
    ) -> None:
        super().__init__(tasks=tasks, split=split)

    def _convert_pdf_to_img(self):
        '''
        Converts pdfs into images
        '''
        dir_path = f"{self.CACHE_DIR}/{self.dataset_name}"
        for fn in os.listdir(f"{dir_path}/pdfs"): 
            if fn.endswith(".pdf"):
                images = convert_from_path(
                    pdf_path=f"{dir_path}/pdfs/{fn}",
                    dpi=200,
                    fmt="jpg"
                )
                images[0].save(f"{dir_path}/pdfs/{fn.strip('.pdf')}.jpg")
                os.remove(f"{dir_path}/pdfs/{fn}")


    def _download(self) -> None:
        raise Exception("Download DocILE dataset before loading it")

    def _load_data(self) -> None:
        dir_path = f"{self.CACHE_DIR}/{self.dataset_name}"

        self._convert_pdf_to_img()
        split_file: list[str] = json.load(open(f"{dir_path}/{self.split}.json", "r"))
        split_file.sort()

        for img_fn in split_file:
            fields, entities = [], []

            if Task.OCR in self.tasks:
                label: dict = json.load(open(f"{dir_path}/ocr/{img_fn}.json", "r"))
                if os.path.exists(f"{dir_path}/pdfs/{img_fn}.jpg"):
                    img = Image.open(f"{dir_path}/pdfs/{img_fn}.jpg")
                    for block in label["pages"][0]["blocks"]:
                        for line in block["lines"]:
                            for word in line["words"]:
                                x1, y1 = word["geometry"][0]
                                x2, y2 = word["geometry"][1]

                                fields.append(self._convert_to_format(
                                    task = Task.OCR,
                                    item = dict(
                                        bbox = [int(x1 * img.width), int(y1 * img.height), int(x2 * img.width), int(y2 * img.height)],
                                        text = word["value"]
                                    )
                                ))

            if Task.KIE in self.tasks:
                label: dict = json.load(open(f"{dir_path}/annotations/{img_fn}.json", "r"))
                if os.path.exists(f"{dir_path}/pdfs/{img_fn}.jpg"):
                    for extraction in label["field_extractions"]:
                        entities.append(self._convert_to_format(
                            task = Task.KIE,
                            item = dict(
                                label = extraction["fieldtype"],
                                value = extraction["text"]
                            )
                        ))

            if len(fields) > 0 or len(entities) > 0:
                self.data.append(Data(
                    image_path=f"{dir_path}/pdfs/{img_fn}.jpg",
                    fields=fields,
                    entities=entities
                ))
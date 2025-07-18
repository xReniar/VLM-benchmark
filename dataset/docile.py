from .ds import Dataset, Data, Task
from pdf2image import convert_from_path
from PIL import Image
import json
import os


class DocILE(Dataset):
    def __init__(
        self,
        task: Task,
        split: str
    ) -> None:
        super().__init__(task=task, split=split)
        self._convert_pdf_to_img()
        self._load_data()

    def _convert_pdf_to_img(self):
        '''
        Converts pdfs into images
        '''
        for fn in os.listdir(f"./data/docile/pdfs"): 
            if fn.endswith(".pdf"):
                images = convert_from_path(
                    pdf_path=f"./data/docile/pdfs/{fn}",
                    dpi=200,
                    fmt="jpg"
                )
                images[0].save(f"./data/docile/pdfs/{fn.strip('.pdf')}.jpg")
                os.remove(f"./data/docile/pdfs/{fn}")

    def _load_data(self) -> None:
        split_file: list[str] = json.load(open(f"./data/docile/{self.split}.json", "r"))
        split_file.sort()

        folder_name = "ocr" if self.task == Task.OCR else "annotations"

        for img_fn in split_file:
            label: dict = json.load(open(f"./data/docile/{folder_name}/{img_fn}.json", "r"))

            if self.task == Task.OCR:
                fields = []
                img = Image.open(f"./data/docile/pdfs/{img_fn}.jpg")
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

                self.data.append(Data(
                    image_path=f"./data/docile/pdfs/{img_fn}.jpg",
                    fields=fields
                ))

            if self.task == Task.KIE:
                entities = []

                if "field_extractions" in label:
                    for extraction in label["field_extractions"]:
                        entities.append(self._convert_to_format(
                            task = Task.KIE,
                            item = dict(
                                label = extraction["fieldtype"],
                                value = extraction["text"]
                            )
                        ))

                    self.data.append(Data(
                        image_path=f"./data/docile/pdfs/{img_fn}.jpg",
                        entities=entities
                    ))



        
from .ds import Dataset, Data, Task
from pdf2image import convert_from_path
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
            if not fn.endswith(".pdf"):
                images = convert_from_path(
                    pdf_path=f"./data/docile/pdfs/{fn}.pdf",
                    dpi=200,
                    fmt="jpg"
                )
                images[0].save(f"./data/docile/pdfs/{fn}.jpg")
                os.remove(f"./data/docile/pdfs/{fn}.pdf")

    def _load_data(self) -> None:
        split_file: list[str] = json.load(open(f"./data/docile/{self.split}.json", "r"))

        
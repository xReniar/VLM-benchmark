from enum import Enum
from pydantic import BaseModel
import os


class Task(Enum):
    CLS = "cls"
    KIE = "kie"
    OCR = "ocr"
    VQA = "vqa"
    OBJ = "obj"

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    normalized: bool = False

class Field(BaseModel):
    label: str
    value: str
    bbox: BBox | None = None

class VQA(BaseModel):
    question: str
    answer: str

class Classification(BaseModel):
    doc_type: str
    labels: list[str]

class Data(BaseModel):
    image_path: str
    fields: list[Field] | None
    entities: list[Field] | None
    objects: list[Field] | None
    vqa: list[VQA] | None
    cls: Classification | None

class Dataset(BaseModel):
    data: list[Data]
    task: Task
    split: str
    dir: str | None = None

    def read_folder(self, path: str) -> list[str]:
        folder = os.listdir(path)
        folder.sort()
        return folder

    def _convert_to_format(self, task: Task, item: dict) -> Field | VQA | Classification:
        '''
        This functions converts the `item` into "Field", "VQA" or "Classification" type based on the `task` parameter
        '''
        processed = None

        if task == Task.CLS:
            processed = Classification(
                doc_type = item["doc_type"],
                labels = item["labels"]
            )
        elif task == Task.KIE:
            processed = Field(
                label = item["label"],
                value = item["value"],
                bbox = None
            )
        elif task == Task.OCR:
            x1, y1, x2, y2 = tuple(item["bbox"])
            processed = Field(
                label = "text",
                value = item["text"],
                bbox = BBox(
                    x1, y1, x2, y2,
                    item["normalized"])
            )
        elif task == Task.VQA:
            processed = VQA(
                question = item["question"],
                answer = item["answer"]
            )
        elif task == Task.OBJ:
            processed = Field(
                label = "object",
                value = None,
                bbox = BBox()
            )
        else:
            raise Exception(f"Task {task} does not exist")

        return processed
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
    task: Task
    fields: list[Field] | None = None
    entities: list[Field] | None = None
    objects: list[Field] | None = None
    vqa: list[VQA] | None = None
    cls: Classification | None = None

class Dataset(BaseModel):
    task: Task
    split: str
    data: list[Data] = []

    def read_folder(self, path: str) -> list[str]:
        folder = os.listdir(path)
        folder.sort()
        return folder
    
    def __iter__(self):
        for item in self.data:
            if self.task == Task.CLS:
                yield {"image_path": item.image_path, "cls": item.cls}
            elif self.task == Task.KIE:
                yield {"image_path": item.image_path, "entities": item.entities}
            elif self.task == Task.OCR:
                yield {"image_path": item.image_path, "fields": item.fields}
            elif self.task == Task.VQA:
                yield {"image_path": item.image_path, "vqa": item.vqa}
            elif self.task == Task.OBJ:
                yield {"image_path": item.image_path, "objects": item.objects}
            else:
                raise ValueError(f"Task {self.task} not supported")

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
                bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
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
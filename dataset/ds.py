from enum import Enum
from pydantic import BaseModel


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
    text: list[Field] | None
    objects: list[Field] | None
    vqa: list[VQA] | None
    cls: Classification | None

class Dataset(BaseModel):
    task: Task
    split: str
    dir: str | None = None

    def _convert_to_format(self, task: Task, item) -> Field | VQA | Classification:
        '''
        This functions converts the `item` into "Field", "VQA" or "Classification" type based on the `task` parameter
        '''
        processed = None

        if task == Task.CLS:
            processed = Classification(
                doc_type = None,
                labels = []
            )
        elif task == Task.KIE:
            processed = Field(
                label = None,
                value = None,
                bbox = None
            )
        elif task == Task.OCR:
            processed = Field(
                label = "text",
                value = None,
                bbox = BBox()
            )
        elif task == Task.VQA:
            processed = VQA(
                question = None,
                answer = None
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
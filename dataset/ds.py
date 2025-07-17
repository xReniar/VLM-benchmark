from enum import Enum
from pydantic import BaseModel


class Task(Enum):
    CLS = "cls"
    KIE = "kie"
    OCR = "ocr"
    VQA = "vqa"
    OBJDE = "objde"

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
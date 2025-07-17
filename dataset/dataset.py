from enum import Enum
from pydantic import BaseModel


class DatasetType(Enum):
    CLS = "cls"
    KIE = "kie"
    OCR = "ocr"
    VQA = "vqa"

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Text(BaseModel):
    text: str
    bbox: BBox | None = None

class Field(BaseModel):
    label: str
    value: Text

class VQA(BaseModel):
    question: str
    answer: str

class CLS(BaseModel):
    doc_type: str
    labels: list[str]
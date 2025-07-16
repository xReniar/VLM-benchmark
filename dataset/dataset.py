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

class Field(BaseModel):
    label: str
    value: str | list[str]
    bbox: BBox | None = None

class Dataset(BaseModel):
    image_path: list[str]

class CLS(BaseModel):
    image_path: str
    category: str
    text: str
    bbox: BBox

class KIE(BaseModel):
    image_path: str
    text: str
    entity: str
    bbox: BBox

class OCR(BaseModel):
    image_path: str
    text: str
    bbox: BBox

class VQA(BaseModel):
    image_path: str
    question: str
    answer: str | list[str]
    bbox: BBox

class OCRDataset(BaseModel):
    images_path: list[str]
    


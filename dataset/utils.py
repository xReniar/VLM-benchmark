from __future__ import annotations

import json
import os

from pdf2image import convert_from_path


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def convert_pdf2image(pdf_path: str, output_dir: str):
    """
    Convert a pdf file to a list of image files.
    Args:
        pdf_path: str, path to the pdf file. eg: "document.pdf"
        output_dir: str, path to the output directory. eg: "document_images"
    Returns:
        save_paths: list, paths to the image files. eg: ["document_images/document_0.jpeg", "document_images/document_1.jpeg", ...]
    """
    images = convert_from_path(pdf_path)
    base_filename = os.path.basename(pdf_path)
    save_paths = []
    for i, image in enumerate(images):
        save_path = os.path.join(output_dir, f"{base_filename}_{i}.jpeg")
        image.save(save_path, "JPEG")
        save_paths.append(save_path)
    return save_paths
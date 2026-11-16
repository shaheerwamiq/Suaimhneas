"""PDF page extraction and simple horizontal text-line segmentation."""

import re
from typing import Optional

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image

from .runtime import TrocrEngine


def segment_text_lines(image: Image.Image) -> list[Image.Image]:
    """Group nearby printed glyphs by baseline and return top-to-bottom crops."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    width, height = image.size
    kernel_width = max(9, min(35, width // 70))
    joined = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    components = []
    minimum_height = max(4, height // 500)
    for x, y, box_width, box_height, area in stats[1:count]:
        if box_height >= minimum_height and area >= 8:
            components.append((int(x), int(y), int(box_width), int(box_height)))
    components.sort(key=lambda box: (box[1] + box[3] / 2, box[0]))

    rows = []
    for x, y, box_width, box_height in components:
        center = y + box_height / 2
        row = next(
            (candidate for candidate in reversed(rows)
             if abs(center - candidate["center"]) <= max(box_height, candidate["height"]) * 0.6),
            None,
        )
        if row is None:
            rows.append({"left": x, "top": y, "right": x + box_width,
                         "bottom": y + box_height, "center": center, "height": box_height})
        else:
            row["left"] = min(row["left"], x)
            row["top"] = min(row["top"], y)
            row["right"] = max(row["right"], x + box_width)
            row["bottom"] = max(row["bottom"], y + box_height)
            row["center"] = (row["top"] + row["bottom"]) / 2
            row["height"] = row["bottom"] - row["top"]

    rows.sort(key=lambda row: (row["top"], row["left"]))
    crops = []
    for row in rows:
        pad_x = max(4, row["height"] // 2)
        pad_y = max(3, row["height"] // 4)
        box = (max(0, row["left"] - pad_x), max(0, row["top"] - pad_y),
               min(width, row["right"] + pad_x), min(height, row["bottom"] + pad_y))
        crops.append(image.crop(box).convert("RGB"))
    return crops or [image.convert("RGB")]


def read_pdf(path: str, engine: Optional[TrocrEngine] = None, dpi: int = 180) -> str:
    """Extract selectable PDF text directly and OCR image-only pages offline."""
    pages = []
    ocr_engine = engine
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if len(re.sub(r"\W", "", text)) >= 20:
                pages.append(f"--- Page {page_number} ---\n{text}")
                continue

            scale = dpi / 72
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            page_image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            if ocr_engine is None:
                ocr_engine = TrocrEngine()
            lines = [ocr_engine.recognize(line) for line in segment_text_lines(page_image)]
            page_text = "\n".join(line for line in lines if line)
            pages.append(f"--- Page {page_number} ---\n{page_text}".rstrip())
    return "\n\n".join(pages)
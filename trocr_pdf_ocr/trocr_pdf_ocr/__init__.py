"""Offline PDF OCR with a packaged TrOCR ONNX model."""

from .pdf import read_pdf
from .runtime import TrocrEngine

__all__ = ["TrocrEngine", "read_pdf"]
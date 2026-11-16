"""Command-line entry point for offline PDF OCR."""

import argparse
from pathlib import Path

from .pdf import read_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from a PDF using local TrOCR ONNX assets.")
    parser.add_argument("pdf", type=Path, help="Path to a local PDF file")
    parser.add_argument("-o", "--output", type=Path, help="Write extracted text to this file")
    parser.add_argument("--dpi", type=int, default=180, help="Render resolution for scanned pages (default: 180)")
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF does not exist: {args.pdf}")
    text = read_pdf(str(args.pdf), dpi=args.dpi)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
import io
import tempfile
import unittest
from pathlib import Path

import pymupdf
import numpy as np
from PIL import Image, ImageDraw
from unittest.mock import patch

from trocr_pdf_ocr.pdf import read_pdf, segment_text_lines
from trocr_pdf_ocr.runtime import _materialize_bundled_model


class PdfOcrTests(unittest.TestCase):
    def test_text_pdf_is_extracted_without_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "text.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "This selectable text is extracted directly.")
            document.save(path)
            document.close()

            result = read_pdf(str(path))

        self.assertIn("This selectable text is extracted directly.", result)

    def test_scanned_page_line_segmentation_is_top_to_bottom(self):
        image = Image.new("RGB", (600, 240), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 40, 260, 55), fill="black")
        draw.rectangle((30, 120, 420, 135), fill="black")

        lines = segment_text_lines(image)

        self.assertEqual(len(lines), 2)
        self.assertGreater(np.asarray(lines[1]).shape[0], 0)

    def test_scanned_pdf_uses_ocr_engine(self):
        class FakeEngine:
            def recognize(self, image):
                return "recognized scan line"

        image = Image.new("RGB", (600, 240), "white")
        ImageDraw.Draw(image).rectangle((30, 40, 260, 55), fill="black")
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.pdf"
            document = pymupdf.open()
            page = document.new_page(width=600, height=240)
            page.insert_image(page.rect, stream=image_bytes.getvalue())
            document.save(path)
            document.close()

            result = read_pdf(str(path), engine=FakeEngine(), dpi=72)

        self.assertIn("recognized scan line", result)

    def test_model_chunks_reassemble_and_verify(self):
        import hashlib
        import json

        with tempfile.TemporaryDirectory() as temp_dir:
            package_model = Path(temp_dir) / "package-model"
            cache_dir = Path(temp_dir) / "cache"
            package_model.mkdir()
            config = b"local config"
            weight = b"chunked model weights"
            (package_model / "config.json").write_bytes(config)
            (package_model / "weights.onnx.part000").write_bytes(weight[:8])
            (package_model / "weights.onnx.part001").write_bytes(weight[8:])
            hashes = {
                "config.json": hashlib.sha256(config).hexdigest(),
                "weights.onnx": hashlib.sha256(weight).hexdigest(),
            }
            identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            manifest = {
                "files": {"weights.onnx": ["weights.onnx.part000", "weights.onnx.part001"]},
                "file_sha256": hashes,
                "sha256": identity,
            }
            (package_model / "shards.json").write_text(json.dumps(manifest), encoding="utf-8")

            with patch("trocr_pdf_ocr.runtime.PACKAGE_MODEL_DIR", package_model):
                result = _materialize_bundled_model(cache_dir)
            self.assertEqual((result / "weights.onnx").read_bytes(), weight)


if __name__ == "__main__":
    unittest.main()
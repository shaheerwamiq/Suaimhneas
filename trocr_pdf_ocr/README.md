# Offline TrOCR PDF OCR

This package extracts selectable text directly from PDFs and uses a local
ONNX Runtime model for scanned pages. It makes no model-hub or other network
requests during inference.

The default checkpoint is `microsoft/trocr-base-printed`, fine-tuned on SROIE.
It recognizes printed text lines, not full-page layouts or handwriting. OCR
quality can vary on multi-column pages, tables, low-resolution scans, and
documents outside its training domain. PDFs with an embedded text layer do
not invoke the model.

## Install

Clone this repository on the connected machine, then install the package from
the local checkout:

```bash
python -m pip install ./trocr_pdf_ocr
```

The Python dependencies must already be available from your corporate package
mirror or local wheelhouse. The model itself is included in this repository;
inference uses only local files. Model chunks are each below GitHub's 100 MiB
per-file limit and are reassembled into `~/.cache/trocr-pdf-ocr` on first use.

## Use

```bash
trocr-pdf-ocr ./document.pdf -o ./document.txt
```

Or from Python:

```python
from trocr_pdf_ocr import read_pdf

text = read_pdf("document.pdf")
```

## Handwritten Text Model

The optional `microsoft/trocr-base-handwritten` ONNX export is stored beside
the printed model. Select it explicitly when recognizing a text-line image:

```python
from pathlib import Path
from PIL import Image
from trocr_pdf_ocr import TrocrEngine

model_dir = Path("trocr_pdf_ocr/trocr_pdf_ocr/model_handwritten")
engine = TrocrEngine(model_dir=model_dir)
text = engine.recognize(Image.open("handwriting-line.png"))
```

The model expects a cropped, single text line, just like the printed model.
Both model directories remain separate; the default PDF reader continues to
use the printed checkpoint unless an engine is passed explicitly. To OCR
scanned PDF pages with the handwritten checkpoint, pass it to the PDF reader:

```python
from pathlib import Path
from trocr_pdf_ocr import TrocrEngine, read_pdf

engine = TrocrEngine(
	model_dir=Path("trocr_pdf_ocr/trocr_pdf_ocr/model_handwritten")
)
text = read_pdf("handwritten-scan.pdf", engine=engine)
```

## Export / Rebuild The Model

The repository includes a verified model bundle. To rebuild it from the
official checkpoint on a machine with model-host access, install
`trocr_pdf_ocr/requirements-export.txt` and run:

```bash
python -m pip install -r trocr_pdf_ocr/requirements-export.txt
python -m trocr_pdf_ocr.export_model
```

This exports a CPU ONNX encoder and decoder, dynamically quantizes their
weights to INT8, saves the processor/configuration files, and splits large
files into standard-Git-safe chunks. Keep the generated contents of
`trocr_pdf_ocr/trocr_pdf_ocr/model/` when publishing the repository.

See [MODEL_LICENSE.txt](MODEL_LICENSE.txt) before redistributing the checkpoint.
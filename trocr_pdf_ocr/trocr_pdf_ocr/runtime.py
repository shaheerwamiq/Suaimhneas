"""Local-only ONNX Runtime execution for TrOCR printed-text checkpoints."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from transformers import TrOCRProcessor


PACKAGE_MODEL_DIR = Path(__file__).resolve().parent / "model"


def _materialize_bundled_model(
    cache_dir: Optional[Path] = None,
    package_model_dir: Path = PACKAGE_MODEL_DIR,
) -> Path:
    """Reassemble GitHub-safe model chunks into a local cache directory."""
    manifest_path = package_model_dir / "shards.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "TrOCR model assets are missing. Install the complete repository package "
            "including its trocr_pdf_ocr/model files."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_hash = manifest["sha256"]
    cache_root = cache_dir or Path(
        os.environ.get("TROCR_CACHE_DIR", Path.home() / ".cache" / "trocr-pdf-ocr")
    )
    target = cache_root / model_hash
    marker = target / ".complete"
    if marker.is_file() and marker.read_text(encoding="ascii") == model_hash:
        return target

    cache_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{model_hash}.", dir=cache_root))
    try:
        chunk_names = {chunk for chunks in manifest["files"].values() for chunk in chunks}
        for source in package_model_dir.iterdir():
            if source.is_file() and source.name not in chunk_names and source.name != "shards.json":
                shutil.copy2(source, staging / source.name)

        for filename, chunks in manifest["files"].items():
            digest = hashlib.sha256()
            with (staging / filename).open("wb") as output:
                for chunk_name in chunks:
                    chunk_path = package_model_dir / chunk_name
                    if not chunk_path.is_file():
                        raise FileNotFoundError(f"TrOCR model chunk is missing: {chunk_name}")
                    with chunk_path.open("rb") as chunk:
                        shutil.copyfileobj(chunk, output)
                        chunk.seek(0)
                        for block in iter(lambda: chunk.read(1024 * 1024), b""):
                            digest.update(block)
            if digest.hexdigest() != manifest["file_sha256"][filename]:
                raise ValueError(f"TrOCR model file failed its SHA-256 check: {filename}")

        for filename, expected_hash in manifest["file_sha256"].items():
            if filename in manifest["files"]:
                continue
            digest = hashlib.sha256()
            with (staging / filename).open("rb") as file:
                for block in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"TrOCR model file failed its SHA-256 check: {filename}")

        (staging / ".complete").write_text(model_hash, encoding="ascii")
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


class TrocrEngine:
    """Recognize text-line images using local ONNX files and CPU execution."""

    def __init__(self, model_dir: Optional[os.PathLike] = None, max_new_tokens: int = 128):
        if model_dir is None:
            self.model_dir = _materialize_bundled_model()
        else:
            requested_model_dir = Path(model_dir)
            self.model_dir = (
                _materialize_bundled_model(package_model_dir=requested_model_dir)
                if (requested_model_dir / "shards.json").is_file()
                else requested_model_dir
            )
        if not (self.model_dir / "encoder_model.onnx").is_file():
            raise FileNotFoundError(f"TrOCR encoder ONNX file not found in {self.model_dir}")
        if not (self.model_dir / "decoder_model.onnx").is_file():
            raise FileNotFoundError(f"TrOCR decoder ONNX file not found in {self.model_dir}")

        self.processor = TrOCRProcessor.from_pretrained(
            str(self.model_dir), local_files_only=True, use_fast=False
        )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        thread_count = os.getenv("TROCR_ORT_THREADS")
        if thread_count:
            options.intra_op_num_threads = max(1, int(thread_count))
        self.encoder = ort.InferenceSession(
            str(self.model_dir / "encoder_model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.decoder = ort.InferenceSession(
            str(self.model_dir / "decoder_model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        with (self.model_dir / "generation_config.json").open(encoding="utf-8") as file:
            generation = json.load(file)
        self.start_token_id = generation.get("decoder_start_token_id")
        self.eos_token_id = generation.get("eos_token_id", self.processor.tokenizer.eos_token_id)
        if self.start_token_id is None:
            raise ValueError("generation_config.json has no decoder_start_token_id")
        self.max_new_tokens = max_new_tokens

    def recognize(self, image) -> str:
        """Recognize one cropped text line; no network access is performed."""
        pixel_values = self.processor(
            images=image.convert("RGB"), return_tensors="np"
        ).pixel_values
        encoder_inputs = {self.encoder.get_inputs()[0].name: pixel_values}
        hidden_states = self.encoder.run(None, encoder_inputs)[0]
        token_ids = [int(self.start_token_id)]

        for _ in range(self.max_new_tokens):
            decoder_inputs = {
                self.decoder.get_inputs()[0].name: np.asarray([token_ids], dtype=np.int64),
                self.decoder.get_inputs()[1].name: hidden_states,
            }
            logits = self.decoder.run(None, decoder_inputs)[0]
            next_token = int(np.argmax(logits[0, -1, :]))
            if next_token == self.eos_token_id:
                break
            token_ids.append(next_token)

        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
"""Export TrOCR to CPU-friendly INT8 ONNX files without Optimum."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MAX_GIT_FILE_BYTES = 90 * 1024 * 1024
MODEL_ID = "microsoft/trocr-base-printed"
GENERATED_FILES = {
    "encoder_model.onnx", "decoder_model.onnx", "config.json", "generation_config.json",
    "preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "tokenizer.json", "shards.json",
}


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values):
        return self.encoder(pixel_values=pixel_values, return_dict=False)[0]


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, decoder_input_ids, encoder_hidden_states):
        return self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=False,
        )[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_large_files(output_dir: Path) -> dict:
    shards = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "shards.json" or path.stat().st_size <= MAX_GIT_FILE_BYTES:
            continue
        file_hash = _sha256(path)
        chunks = []
        with path.open("rb") as source:
            index = 0
            while block := source.read(MAX_GIT_FILE_BYTES):
                chunk_name = f"{path.name}.part{index:03d}"
                (output_dir / chunk_name).write_bytes(block)
                chunks.append(chunk_name)
                index += 1
        shards[path.name] = {"chunks": chunks, "sha256": file_hash}
        path.unlink()
    return shards


def export_model(model_id: str = MODEL_ID, output_dir: Path = Path(__file__).parent / "model") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.name in GENERATED_FILES or path.name.startswith((
            "encoder_model.onnx.part", "decoder_model.onnx.part",
            "encoder_model.onnx.", "decoder_model.onnx.",
        )):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    processor = TrOCRProcessor.from_pretrained(model_id, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).eval()
    model.config.use_cache = False
    pixels = processor(images=Image.new("RGB", (384, 384), "white"), return_tensors="pt").pixel_values
    start_id = model.generation_config.decoder_start_token_id
    if start_id is None:
        raise ValueError("The checkpoint has no decoder_start_token_id")
    decoder_ids = torch.tensor([[start_id]], dtype=torch.long)
    hidden = model.encoder(pixel_values=pixels, return_dict=False)[0]

    with tempfile.TemporaryDirectory(prefix="trocr-export-") as temp_dir:
        temp = Path(temp_dir)
        encoder_path = temp / "encoder_model.onnx"
        decoder_path = temp / "decoder_model.onnx"
        torch.onnx.export(
            Encoder(model), (pixels,), str(encoder_path),
            input_names=["pixel_values"], output_names=["encoder_hidden_states"],
            dynamic_axes={"pixel_values": {0: "batch_size"},
                          "encoder_hidden_states": {0: "batch_size"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        torch.onnx.export(
            Decoder(model), (decoder_ids, hidden), str(decoder_path),
            input_names=["decoder_input_ids", "encoder_hidden_states"],
            output_names=["logits"],
            dynamic_axes={"decoder_input_ids": {0: "batch_size", 1: "sequence_length"},
                          "encoder_hidden_states": {0: "batch_size"},
                          "logits": {0: "batch_size", 1: "sequence_length"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        quantize_dynamic(str(encoder_path), str(output_dir / "encoder_model.onnx"),
                         weight_type=QuantType.QInt8, per_channel=True)
        quantize_dynamic(str(decoder_path), str(output_dir / "decoder_model.onnx"),
                         weight_type=QuantType.QInt8, per_channel=True)

    model.config.save_pretrained(output_dir)
    model.generation_config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    shards = _split_large_files(output_dir)
    file_hashes = {
        path.name: _sha256(path)
        for path in output_dir.iterdir()
        if path.is_file() and ".part" not in path.name and path.name != "shards.json"
    }
    all_hashes = {
        **file_hashes,
        **{name: info["sha256"] for name, info in shards.items()},
    }
    manifest = {
        "model_id": model_id,
        "source_revision": getattr(model.config, "_commit_hash", None),
        "files": {name: info["chunks"] for name, info in shards.items()},
        "file_sha256": all_hashes,
        "sha256": hashlib.sha256(
            json.dumps(all_hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    (output_dir / "shards.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {model_id} to {output_dir}")
    print(f"Files over 90 MiB were split for standard Git hosting: {len(shards)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model id or local Transformers checkpoint")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "model")
    args = parser.parse_args()
    export_model(args.model, args.output)


if __name__ == "__main__":
    main()
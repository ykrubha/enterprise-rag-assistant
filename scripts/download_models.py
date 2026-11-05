"""Download the search models at pinned versions, with no Hugging Face access needed.

    python scripts/download_models.py

Fetches ONNX exports of all-MiniLM-L6-v2 and ms-marco-MiniLM-L-6-v2 (both Apache-2.0, from
sentence-transformers) from an npm package that republishes them. The package isn't run. Only
the listed model files are extracted, and each is checked against a SHA-256 hash.
"""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://registry.npmjs.org/kontexta-reranker-model/-/kontexta-reranker-model-1.1.0.tgz"
ARCHIVE_SHA256 = "36cad4891b0c0c806d945f427099307e14f41fd5241f56581491cfa1ea3f4d33"
PREFIX = "package/models/Xenova"
FILES = {
    "all-MiniLM-L6-v2/onnx/model_quantized.onnx": (
        "all-MiniLM-L6-v2/model_quantized.onnx",
        "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1",
    ),
    "all-MiniLM-L6-v2/tokenizer.json": ("all-MiniLM-L6-v2/tokenizer.json", None),
    "ms-marco-MiniLM-L-6-v2/onnx/model.onnx": (
        "ms-marco-MiniLM-L-6-v2/model.onnx",
        "c623d0bcb99f4622beb413eaef00cfbe5db20df9f1dd982da4b4f26022881870",
    ),
    "ms-marco-MiniLM-L-6-v2/tokenizer.json": ("ms-marco-MiniLM-L-6-v2/tokenizer.json", None),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    target = Path("data/models")
    if all((target / destination).exists() for destination, _ in FILES.values()):
        print("Models already present.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "models.tgz"
        print("Downloading models (about 100 MB)...")
        urllib.request.urlretrieve(URL, archive)
        if sha256(archive) != ARCHIVE_SHA256:
            raise SystemExit("Checksum mismatch for the model archive.")
        with tarfile.open(archive) as tar:
            for source, (destination, expected) in FILES.items():
                path = target / destination
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(f"{PREFIX}/{source}") as stream:
                    path.write_bytes(stream.read())
                if expected and sha256(path) != expected:
                    path.unlink()
                    raise SystemExit(f"Checksum mismatch for {destination}.")
    print(f"Models saved to {target}.")


if __name__ == "__main__":
    main()

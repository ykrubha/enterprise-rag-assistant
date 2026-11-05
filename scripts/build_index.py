"""Download the Kubernetes concept docs at a pinned commit and build the search index.

    python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from kb_assistant.index import HybridIndex, MiniLmEmbedder
from kb_assistant.ingest import load_chunks

REPO = "https://github.com/kubernetes/website.git"
COMMIT = "8f6d9762bfd9e130d63f678c2e0f9b048bc3373e"
DOCS = "content/en/docs/concepts"


def fetch_corpus(target: Path) -> Path:
    docs = target / DOCS
    if docs.exists():
        return docs
    target.mkdir(parents=True, exist_ok=True)
    commands = [
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", REPO],
        ["git", "config", "remote.origin.promisor", "true"],
        ["git", "config", "remote.origin.partialclonefilter", "blob:none"],
        ["git", "sparse-checkout", "set", DOCS],
        ["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT],
        ["git", "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"],
    ]
    for command in commands:
        subprocess.run(command, cwd=target, check=True)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="data/k8s-website")
    parser.add_argument("--docs", default=None, help="Use this Markdown folder instead of downloading.")
    parser.add_argument("--index-dir", default="data/index")
    args = parser.parse_args()

    docs = Path(args.docs) if args.docs else fetch_corpus(Path(args.corpus_dir))
    started = time.perf_counter()
    chunks = load_chunks(docs)
    index = HybridIndex(chunks, MiniLmEmbedder())
    index.save(args.index_dir)
    documents = len({c.doc_id for c in chunks})
    print(f"Indexed {documents} documents into {len(chunks)} chunks "
          f"in {time.perf_counter() - started:.1f}s -> {args.index_dir}")


if __name__ == "__main__":
    main()

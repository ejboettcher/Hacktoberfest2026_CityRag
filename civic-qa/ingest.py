"""Build a search index from public-meeting records: PDF, .txt and .md files.

Every chunk keeps a location ("p. 4" for PDFs, "lines 12-30" for text files)
so answers can cite exactly where they came from.
"""
import json
import pathlib
import sys

import numpy as np
import ollama
from pypdf import PdfReader

EMBED_MODEL = "nomic-embed-text"
CHUNK_CHARS = 1200   # roughly 250 words per chunk
BATCH = 32
TEXT_TYPES = {".txt", ".md"}


def chunks_from_pdf(path):
    """Yield page-tagged text chunks so every answer can cite a page."""
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        for i in range(0, len(text), CHUNK_CHARS):
            piece = text[i:i + CHUNK_CHARS]
            if len(piece) > 50:
                yield {"source": path.name, "loc": f"p. {page_num}", "text": piece}


def chunks_from_text(path):
    """Yield line-tagged chunks from a .txt or .md file.

    Splits on blank lines (paragraphs, Markdown sections) and packs
    paragraphs together up to CHUNK_CHARS, so related sentences stay together.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    buf, start = [], None

    def flush(end):
        text = " ".join(" ".join(buf).split())
        if len(text) > 50:
            yield {"source": path.name, "loc": f"lines {start}-{end}", "text": text}

    for n, line in enumerate(lines, start=1):
        if start is None and line.strip():
            start = n
        # A new Markdown heading, or a full buffer, starts a new chunk.
        new_section = line.lstrip().startswith("#") and buf
        too_long = sum(len(b) for b in buf) + len(line) > CHUNK_CHARS and buf
        if new_section or too_long:
            yield from flush(n - 1)
            buf, start = [], n
        if line.strip():
            buf.append(line.strip())
    if buf:
        yield from flush(len(lines))


def chunks_from_file(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return chunks_from_pdf(path)
    if suffix in TEXT_TYPES:
        return chunks_from_text(path)
    return iter(())


def main(folder):
    files = sorted(p for p in pathlib.Path(folder).iterdir()
                   if p.suffix.lower() in TEXT_TYPES | {".pdf"})
    chunks = [c for f in files for c in chunks_from_file(f)]
    if not chunks:
        sys.exit(f"No readable text found in {folder}. Scanned PDFs need ocr.py first.")
    vectors = []
    for i in range(0, len(chunks), BATCH):
        batch = [f"search_document: {c['text']}" for c in chunks[i:i + BATCH]]
        vectors.extend(ollama.embed(model=EMBED_MODEL, input=batch)["embeddings"])
        print(f"Embedded {min(i + BATCH, len(chunks))}/{len(chunks)}")
    np.save("index.npy", np.array(vectors, dtype=np.float32))
    pathlib.Path("chunks.json").write_text(json.dumps(chunks))
    print(f"Indexed {len(chunks)} chunks from {len(files)} files")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs")

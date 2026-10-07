"""Build a search index from public-meeting records: PDF, .txt and .md files.

Pass one or more folders, e.g. `ingest.py docs docs_ocr`. Folders are read in
order and chunks whose text already appeared are indexed once, so the readable
copy in docs/ wins over the same minutes OCR'd into docs_ocr/. Every chunk
keeps a location ("p. 4" for PDFs, "lines 12-30" for text files) so answers
can cite exactly where they came from.

Unlike the original embedding-based civic-qa, this builds a BM25 lexical
index (rank_bm25) instead of a vector index: no embedding model is pulled,
nothing is embedded, and ingest takes seconds instead of minutes.
Semantic gaps are handled at query time by the clef-flash reranker in
ask.py plus the synonym expansion in glossary.py.
"""

import hashlib
import json
import pathlib
import pickle
import re
import sys

from pypdf import PdfReader
from rank_bm25 import BM25Okapi

CHUNK_CHARS = 1200  # roughly 250 words per chunk
TEXT_TYPES = {".txt", ".md"}

# BM25 matches exact tokens, so "allocate" would miss "allocation" and
# "parks" would miss "park". Lowercase, strip punctuation, and normalize
# common suffixes; the same tokenizer runs on the corpus and on queries.
_TOKEN = re.compile(r"\w+", re.UNICODE)
_SUFFIXES = ("ations", "ation", "ates", "ate", "ing", "ed", "es", "s")


def _stem(token):
    """Strip a common suffix when at least four characters remain."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text):
    return [_stem(t) for t in _TOKEN.findall(text.lower())]


def chunks_from_pdf(path):
    """Yield page-tagged text chunks so every answer can cite a page."""
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        for i in range(0, len(text), CHUNK_CHARS):
            piece = text[i : i + CHUNK_CHARS]
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


def collect(folders):
    """List indexable files from each folder, in order, without recursing."""
    files = []
    for folder in folders:
        path = pathlib.Path(folder)
        if not path.is_dir():
            sys.exit(
                f"{folder} is not a folder. Pass the folders to index, "
                "e.g. `uv run python ingest.py docs docs_ocr`."
            )
        files.extend(
            sorted(
                p for p in path.iterdir() if p.suffix.lower() in TEXT_TYPES | {".pdf"}
            )
        )
    return files


def dedupe(chunks):
    """Keep the first chunk for each distinct text, ignoring case and spacing.

    The same minutes can appear twice (a readable copy in docs/ plus a scan
    OCR'd into docs_ocr/); folders are read in argument order, so the first
    copy survives and later duplicates are dropped.
    """
    seen, kept = set(), []
    for c in chunks:
        key = hashlib.sha1(" ".join(c["text"].lower().split()).encode()).digest()
        if key not in seen:
            seen.add(key)
            kept.append(c)
    return kept


def main(*folders):
    files = collect(folders)
    chunks = [c for f in files for c in chunks_from_file(f)]
    unique = dedupe(chunks)
    if not unique:
        sys.exit(
            "No readable text found. Scanned PDFs need ocr.py first, "
            "then index the output folder too (`ingest.py docs docs_ocr`)."
        )
    corpus = [tokenize(c["text"]) for c in unique]
    bm25 = BM25Okapi(corpus)
    with open("bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    pathlib.Path("chunks.json").write_text(json.dumps(unique))
    skipped = len(chunks) - len(unique)
    note = f", skipped {skipped} duplicate chunks" if skipped else ""
    print(
        f"Indexed {len(unique)} chunks from {len(files)} files "
        f"(BM25, no embeddings){note}"
    )


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["docs"]))

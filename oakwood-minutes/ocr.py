"""Add a searchable text layer to scanned PDFs so ingest.py can read them.

Reads every PDF in the source folder (default docs/) and writes an OCR'd
copy to the output folder (default docs_ocr/). Pages that already have text
are left alone, so it's safe to run on a mix of scanned and digital PDFs.
Re-runs skip an output that is already newer than its source, and a source
whose content duplicates one already processed in the same run.
"""

import hashlib
import pathlib
import sys

import ocrmypdf
from pypdf import PdfReader


def ocr_folder(src="docs", dst="docs_ocr"):
    out_dir = pathlib.Path(dst)
    out_dir.mkdir(exist_ok=True)
    seen = {}
    for pdf in sorted(pathlib.Path(src).glob("*.pdf")):
        digest = hashlib.sha1(pdf.read_bytes()).digest()
        if digest in seen:
            print(f"OCR: {pdf.name} (skipped, same content as {seen[digest]})")
            continue
        seen[digest] = pdf.name
        out = out_dir / pdf.name
        if out.exists() and out.stat().st_mtime >= pdf.stat().st_mtime:
            print(f"OCR: {pdf.name} (skipped, {out} is up to date)")
            continue
        print(f"OCR: {pdf.name}")
        ocrmypdf.ocr(
            pdf,
            out,
            language="eng",
            skip_text=True,
            deskew=True,
            rotate_pages=True,
            progress_bar=False,
        )
        check(out)


def check(pdf):
    """Flag pages where OCR found little text, so a person can look at them."""
    for n, page in enumerate(PdfReader(pdf).pages, start=1):
        chars = len((page.extract_text() or "").strip())
        if chars < 200:
            print(f"  check {pdf.name} p. {n}: only {chars} characters found")


if __name__ == "__main__":
    ocr_folder(*sys.argv[1:])

"""Add a searchable text layer to scanned PDFs so ingest.py can read them.

Reads every PDF in docs/, writes an OCR'd copy to docs_ocr/.
Pages that already have text are left alone, so it's safe to run on
a mix of scanned and digital PDFs.
"""
import pathlib
import sys

import ocrmypdf
from pypdf import PdfReader


def ocr_folder(src="docs", dst="docs_ocr"):
    out_dir = pathlib.Path(dst)
    out_dir.mkdir(exist_ok=True)
    for pdf in sorted(pathlib.Path(src).glob("*.pdf")):
        out = out_dir / pdf.name
        print(f"OCR: {pdf.name}")
        ocrmypdf.ocr(pdf, out, language="eng", skip_text=True,
                     deskew=True, rotate_pages=True, progress_bar=False)
        check(out)


def check(pdf):
    """Flag pages where OCR found little text, so a person can look at them."""
    for n, page in enumerate(PdfReader(pdf).pages, start=1):
        chars = len((page.extract_text() or "").strip())
        if chars < 200:
            print(f"  check {pdf.name} p. {n}: only {chars} characters found")


if __name__ == "__main__":
    ocr_folder(*sys.argv[1:])

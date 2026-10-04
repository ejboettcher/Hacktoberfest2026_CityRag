# civic-qa: Ask Your City Council

Code from the Hacktoberfest 2026 tutorial "Open Models for Civic Good". A local open model (Gemma 4 via Ollama) answers questions about public meeting minutes, with page citations.

## Setup

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
uv sync                      # or: pip install -r requirements.txt
```

OCR also needs Tesseract and Ghostscript (`brew install ocrmypdf` or `sudo apt install ocrmypdf`).

## Scripts, in order

| Script | What it does |
| --- | --- |
| `ocr.py` | Adds a text layer to scanned PDFs in `docs/`, writes to `docs_ocr/` |
| `ingest.py` | Splits PDFs into page-tagged chunks and builds the search index |
| `ask.py` | Answers a question from the command line, with citations |
| `app.py` | Web chat UI for residents (Gradio) |
| `trends.py` | 311 requests vs. budget: year-over-year change and cost per request |
| `dayton_calls.py` | Worked example: trends in Dayton police calls for service |

```bash
uv run python ocr.py
uv run python ingest.py docs_ocr
uv run python ask.py "What did council decide about the pickleball court?"
uv run python app.py
```

Always check answers against the original records. This is not an official city service.

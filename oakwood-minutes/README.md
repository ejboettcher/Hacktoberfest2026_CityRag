# oakwood-minutes: Ask Oakwood City Council — decision-first RAG

A proof-of-concept variant of the Hacktoberfest 2026 tutorial "Open Models for
Civic Good" (`civic-qa`), built over the City of Oakwood, Ohio's council records
and rebuilt around a **decision model** instead of embeddings. It is additive:
`civic-qa` is untouched, and this is not a replacement for it. Local open models
(Ollama) answer questions about public meeting minutes, with page citations.

## How it differs from the original civic-qa

| | Original (`civic-qa`) | This project (`oakwood-minutes`) |
|---|---|---|
| Retrieval | `nomic-embed-text` embeddings + numpy vector index | BM25 lexical index (`rank_bm25`), no embedding model |
| Rerank / gate | none — top-5 by cosine similarity | `clef-flash` decision model (`/v1/systemone`): per-chunk relevance + "is this answerable?" gate in one call |
| Unanswerable questions | slow generation that ends in a refusal | instant "I couldn't find that" from the gate |
| Paraphrase handling | embedding similarity | civic synonym expansion (`glossary.py`) |
| Generation | Gemma 4 via Ollama | unchanged (Gemma 4 via Ollama) |

Pipeline per question:

```
question → BM25 top-8 (ms, no model)
         → clef-flash: 1 Choice ranking every candidate + 1 noul "is this answerable?" (one forward pass)
         → gate < threshold  → instant "I couldn't find that"
         → gate ≥ threshold  → top-5 chunks → Gemma 4 streams cited answer
```

## Setup

```bash
ollama pull gemma4:e4b          # answer generation (default)
ollama pull clef-flash:9b-q8_0  # decision model: rerank + answerability gate (default tag)
uv sync
```

Model names can be overridden with environment variables:

```bash
CIVIC_CHAT_MODEL=gemma4:e2b CIVIC_DECISION_MODEL=clef-flash:9b-q8_0 uv run python ask.py "..."
```

Other knobs: `CIVIC_GATE_THRESHOLD` (default `0.5`).

OCR also needs Tesseract and Ghostscript (`brew install ocrmypdf` or `sudo apt install ocrmypdf`).
`clef-flash` requires Ollama ≥ 0.35.1 and is called via `/v1/systemone`, not chat.

## Scripts, in order

| Script | What it does |
| --- | --- |
| `ocr.py` | Adds a text layer to scanned PDFs, writes to `docs_ocr/`; re-runs skip up-to-date outputs and duplicate scans |
| `ingest.py` | Splits PDFs into page-tagged chunks and builds the BM25 index (`bm25.pkl` + `chunks.json`); accepts several folders and indexes duplicate text once |
| `ask.py` | Answers a question from the command line: BM25 → clef-flash gate/rerank → Gemma answer |
| `glossary.py` | Civic synonym map used to expand queries before BM25 |
| `app.py` | Web chat UI for residents (Gradio) |

```bash
cp -r ../civic-qa/docs docs                               # corpus ships with civic-qa in this repo
uv run python ocr.py docs/NotMachineReadablePDF docs_ocr  # scans -> text layer
uv run python ingest.py docs docs_ocr                     # curated sources first; duplicate text indexed once
uv run python ask.py "What did council decide about the pickleball court?"
uv run python app.py
```

## Test before you demo

Write 5 questions you know the answers to, plus 2 the documents can't
answer. Check:

- every answer cites a real page
- unanswerable questions get "I couldn't find that" (watch stderr for the gate probability)
- no invented names, dates, dollar amounts, or vote counts

Tune `CIVIC_GATE_THRESHOLD` if the gate false-rejects answerable questions.

## Hardware

On a 32 GB machine, `clef-flash` (~8 GB at Q4) and `gemma4:e4b` (~7 GB)
run resident together with room to spare. On 16 GB, use `gemma4:e2b` for
generation. `clef` (27B) fits 32 GB only at 4-bit and is ~5x slower —
use it only if `clef-flash` accuracy disappoints.

Always check answers against the original records. This is not an official city service.

"""Answer questions about the indexed documents, with citations."""
import functools
import json
import pathlib
import sys
import time

import numpy as np
import ollama

CHAT_MODEL = "gemma4:e4b"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 5
MAX_ANSWER_TOKENS = 600     # stop runaway generations
TIMEOUT_SECONDS = 300       # give up instead of hanging forever

SYSTEM = """You help residents understand their local government.
Use ONLY the excerpts provided. Do not use outside knowledge.
Write in plain language at about an 8th-grade reading level.
After each fact, cite its excerpt exactly as labeled, like [minutes-2026-09-15.pdf, p. 4]
or [agenda-notes.md, lines 12-30].
If the excerpts do not answer the question, say:
"I couldn't find that in these records." Then suggest contacting the city clerk.
Never guess names, dates, dollar amounts, or vote counts."""

client = ollama.Client(timeout=TIMEOUT_SECONDS)


def check_setup():
    """Fail fast with a clear message instead of hanging."""
    try:
        installed = {m.model for m in client.list().models}
    except Exception as e:
        sys.exit(f"Can't reach Ollama ({e}). Start it with `ollama serve` or open the Ollama app.")
    for model in (CHAT_MODEL, EMBED_MODEL):
        if not any(name == model or name.startswith(model + ":") for name in installed):
            sys.exit(f"Model {model} isn't installed. Run: ollama pull {model}")
    for f in ("index.npy", "chunks.json"):
        if not pathlib.Path(f).exists():
            sys.exit(f"{f} not found. Run `python ingest.py docs_ocr` first, from this folder.")


@functools.cache
def load_index():
    vectors = np.load("index.npy")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors, json.loads(pathlib.Path("chunks.json").read_text())


def retrieve(question, k=TOP_K):
    vectors, chunks = load_index()
    q = client.embed(model=EMBED_MODEL, input=f"search_query: {question}")["embeddings"][0]
    q = np.array(q, dtype=np.float32)
    scores = vectors @ (q / np.linalg.norm(q))
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]


def stream_answer(question):
    """Yield the answer piece by piece, so text appears as soon as it's ready."""
    hits = retrieve(question)
    context = "\n\n".join(
        f"Excerpt [{h['source']}, {h['loc']}]:\n{h['text']}" for h in hits
    )
    stream = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ],
        think=False,            # Gemma 4 can "think" silently for minutes first; skip it
        stream=True,
        keep_alive="30m",       # keep the model loaded between questions
        options={"temperature": 0.2, "num_predict": MAX_ANSWER_TOKENS},
    )
    for part in stream:
        yield part["message"]["content"]


def answer(question):
    """Return the full answer as one string (used by app.py)."""
    return "".join(stream_answer(question))


if __name__ == "__main__":
    check_setup()
    question = " ".join(sys.argv[1:]) or "What did the council vote on last month?"
    print(f"Searching {len(load_index()[1])} chunks... "
          "(the first question also loads the model, which can take a minute)\n")
    start = time.time()
    for piece in stream_answer(question):
        print(piece, end="", flush=True)
    print(f"\n\n[{time.time() - start:.0f}s]")

"""Answer questions about the indexed documents, with citations.

Decision-first RAG, replacing the original embedding retrieval:

  1. BM25 lexical recall (fast, exact terms) over page-tagged chunks
  2. clef-flash decision model (Ollama /v1/systemone) reranks the
     candidates with one Choice question and gates "is this answerable
     at all?" with one Noul in the same call (semantic_find pattern)
  3. Gemma 4 generates the plain-language cited answer from the
     surviving chunks only

Unanswerable questions get an instant "I couldn't find that" instead of
a slow generation that ends in a refusal.
"""

import functools
import json
import os
import pathlib
import pickle
import sys
import time

import ollama

from glossary import expand
from ingest import tokenize

CHAT_MODEL = os.environ.get("CIVIC_CHAT_MODEL", "gemma4:e4b")
DECISION_MODEL = os.environ.get("CIVIC_DECISION_MODEL", "clef-flash:9b-q8_0")
CANDIDATES = 10  # BM25 recall depth; Ollama systemone Choice caps at 26 options
TOP_K = 8  # chunks sent to the generator after reranking
GATE_THRESHOLD = float(os.environ.get("CIVIC_GATE_THRESHOLD", "0.5"))
MIN_TOP_SCORE = float(os.environ.get("CIVIC_MIN_TOP_SCORE", "3.0"))
SCORE_RATIO = 0.3  # keep chunks scoring >= 30% of the top BM25 hit
MAX_ANSWER_TOKENS = 600  # stop runaway generations
TIMEOUT_SECONDS = 300  # give up instead of hanging forever

SYSTEM = """You help residents understand their local government.
Use ONLY the excerpts provided. Do not use outside knowledge.
Write in plain language at about an 8th-grade reading level.
After each fact, cite its excerpt exactly as labeled, like [minutes-2026-09-15.pdf, p. 4]
or [agenda-notes.md, lines 12-30].
If the excerpts do not answer the question, say:
"I couldn't find that in these records." Then suggest contacting the city clerk.
Never guess names, dates, dollar amounts, or vote counts."""

NOT_FOUND = (
    "I couldn't find that in these records. "
    "Try rephrasing, or contact the city clerk's office for help."
)

client = ollama.Client(timeout=TIMEOUT_SECONDS)


def check_setup():
    """Fail fast with a clear message instead of hanging."""
    try:
        installed = {m.model for m in client.list().models}
    except Exception as e:  # noqa: BLE001 - any connection failure means Ollama is down
        sys.exit(
            f"Can't reach Ollama ({e}). Start it with `ollama serve` or open the Ollama app."
        )
    for model in (CHAT_MODEL, DECISION_MODEL):
        if not any(name == model or name.startswith(model + ":") for name in installed):
            sys.exit(f"Model {model} isn't installed. Run: ollama pull {model}")
    for f in ("bm25.pkl", "chunks.json"):
        if not pathlib.Path(f).exists():
            sys.exit(
                f"{f} not found. Run `uv run python ingest.py docs` first, from this folder."
            )


@functools.cache
def load_index():
    with open("bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    return bm25, json.loads(pathlib.Path("chunks.json").read_text())


def retrieve(question, k=CANDIDATES):
    """BM25 lexical recall with civic synonym expansion.

    Drops weak matches so near-miss text never reaches the decision
    model: keep at most k chunks scoring >= SCORE_RATIO of the top hit,
    and bail out entirely when even the best hit is below MIN_TOP_SCORE
    (nothing in the corpus uses the query's terms). Choice probabilities
    sum to 1 and would otherwise rank junk.
    """
    bm25, chunks = load_index()
    scores = bm25.get_scores(tokenize(expand(question)))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if not ranked or scores[ranked[0]] < MIN_TOP_SCORE:
        return []
    cutoff = scores[ranked[0]] * SCORE_RATIO
    top = [i for i in ranked if scores[i] >= cutoff][:k]
    return [chunks[i] for i in top]


def gate_only(question, hits):
    """Answerability check without ranking (for a single candidate).

    Choice needs 2+ options, so one hit can't be reranked — but it still
    must pass the gate instead of going straight to generation.
    Returns the gate probability.
    """
    state = f"chunk_0| ({hits[0]['source']}, {hits[0]['loc']}):\n{hits[0]['text']}"
    questions = {
        "exists": {
            "type": "noul",
            "instructions": f'Does any excerpt address or answer: "{question}"?',
            "criteria": {
                "true": "At least one excerpt states or directly implies the answer",
                "false": "No excerpt addresses this",
            },
        },
    }
    resp = client.systemone(
        model=DECISION_MODEL, state=state, questions=questions, keep_alive="30m"
    )
    answers = resp["answers"] if isinstance(resp, dict) else resp.answers
    return float(answers["exists"]["noul"])


def rerank_and_gate(question, hits):
    """One clef-flash decision call: Choice rerank + answerability gate.

    Semantic-find pattern: a single Choice question ranks every candidate
    (probabilities sum to 1) while one Noul checks whether any excerpt
    answers the question at all. State is sent once; adding candidates
    only adds Choice options, not more questions.
    Returns (kept_chunks, gate_probability).
    """
    if not hits:
        return [], 0.0
    if len(hits) == 1:
        # Choice needs 2+ options, so rerank is impossible — but the
        # lone hit still faces the answerability gate. A single
        # vaguely-related chunk must not go straight to generation.
        return hits, gate_only(question, hits)
    state = "\n\n".join(
        f"chunk_{i}| ({h['source']}, {h['loc']}):\n{h['text']}"
        for i, h in enumerate(hits)
    )
    questions = {
        "where": {
            "type": "choice",
            "instructions": f'Which excerpt contains the answer to: "{question}"?',
            "criteria": {f"chunk_{i}": None for i in range(len(hits))},
        },
        "exists": {
            "type": "noul",
            "instructions": f'Does any excerpt address or answer: "{question}"?',
            "criteria": {
                "true": "At least one excerpt states or directly implies the answer",
                "false": "No excerpt addresses this",
            },
        },
    }
    resp = client.systemone(
        model=DECISION_MODEL, state=state, questions=questions, keep_alive="30m"
    )
    answers = resp["answers"] if isinstance(resp, dict) else resp.answers
    gate = float(answers["exists"]["noul"])
    probs = answers["where"]["probabilities"]
    ranked = sorted(
        ((float(probs.get(f"chunk_{i}", 0.0)), i) for i in range(len(hits))),
        reverse=True,
    )
    # Choice always ranks something first, so the gate (not the rank)
    # decides unanswerable. Keep top-K unconditionally here.
    kept = [hits[i] for _, i in ranked[:TOP_K]]
    return kept, gate


def stream_answer(question):
    """Yield the answer piece by piece, so text appears as soon as it's ready."""
    hits = retrieve(question)
    kept, gate = rerank_and_gate(question, hits)
    if gate < GATE_THRESHOLD or not kept:
        print(f"[gate: not answerable (p={gate:.2f})]", file=sys.stderr)
        yield NOT_FOUND
        return
    print(
        f"[gate: answerable (p={gate:.2f}), kept {len(kept)}/{len(hits)} chunks]",
        file=sys.stderr,
    )
    context = "\n\n".join(
        f"Excerpt [{h['source']}, {h['loc']}]:\n{h['text']}" for h in kept
    )
    stream = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ],
        think=False,  # Gemma 4 can "think" silently for minutes first; skip it
        stream=True,
        keep_alive="30m",  # keep the model loaded between questions
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
    print(
        f"Searching {len(load_index()[1])} chunks... "
        "(the first question also loads the models, which can take a minute)\n"
    )
    start = time.time()
    for piece in stream_answer(question):
        print(piece, end="", flush=True)
    print(f"\n\n[{time.time() - start:.0f}s]")

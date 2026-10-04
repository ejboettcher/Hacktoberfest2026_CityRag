# Open Models for Civic Good: A Hacktoberfest 2026 Tutorial

Oct 4, 2026 · @evelyn boettcher

By the end of this tutorial you will have a working "Ask Your City Council" tool: an open-weight model, running on your own laptop, that answers questions about public meeting minutes in plain language and cites the page it got each answer from.

Open models suit civic work for three reasons. 

* Public data stays under public control, because nothing leaves your machine. 
* A city, library or nonprofit can run the tool for free, with no per-query API bill. 
* And anyone can inspect, fork and improve the code, which matters when residents need to trust the answers.

This is for developers who know basic Python and Git. You do not need a GPU, ML experience or any paid account. Plan on about 90 minutes for the core build.


## Step 1: Choose an open model

For this tutorial, use **Gemma 4 E4B**: it runs on an ordinary laptop, reads text and images, and is Apache 2.0 licensed. Swap in a bigger model later if your hardware allows.

### Open-weight vs. fully open

Most models called "open source" are really **open-weight**: you can download, run and fine-tune them, but the training data stays private. **Fully open** models also publish the data, training code and checkpoints. For civic projects where residents may ask "how was this trained?", a fully open model gives you a real answer.

### Models to start with

| Model | Download size | Context | License | Openness | Best for |
| --- | --- | --- | --- | --- | --- |
| [Gemma 4 E2B](https://ollama.com/library/gemma4) | 4.6–7.5 GB | 128K | Apache 2.0 | Open-weight | Older laptops, 8 GB RAM |
| [Gemma 4 E4B](https://ollama.com/library/gemma4) (default here) | 6.6–9.5 GB | 128K | Apache 2.0 | Open-weight | Most laptops, 16 GB RAM |
| [Gemma 4 12B](https://ollama.com/library/gemma4) | 7.7–8.0 GB | 256K | Apache 2.0 | Open-weight | Better answers on 16 GB+ machines |
| [Gemma 4 26B](https://ollama.com/library/gemma4) (MoE, \~4B active) | 16–19 GB | 256K | Apache 2.0 | Open-weight | Fast and strong on 32 GB machines |
| [Qwen3.6 27B](https://ollama.com/library/qwen3.6) | 18–19 GB | 256K | Apache 2.0 | Open-weight | Strong multilingual and coding; 24 GB GPU or 32 GB Mac |
| [Olmo 3 7B / 32B](https://allenai.org/olmo) | varies | 65K | Apache 2.0 | Fully open | Projects that need full training transparency |

A few notes on the table:

- Gemma 4 switched from Google's custom terms to Apache 2.0 when it launched in April 2026 ([Ars Technica](https://arstechnica.com/ai/2026/04/google-announces-gemma-4-open-ai-models-switches-to-apache-2-0-license/)), which removes most legal friction for city or nonprofit deployment.
- Gemma 4 E2B and E4B also accept audio input, useful for transcribing public meetings.
- Olmo 3 from Ai2 releases its full "model flow": data, recipe and every training checkpoint ([BusinessWire](https://secure.businesswire.com/news/home/20251120271934/en/Olmo-3-Charting-a-Path-Through-the-Model-Flow-to-Lead-Open-Source-AI)).
- Always read the license before deploying. Llama 4 uses Meta's community license, which carries usage restrictions and is not OSI-approved.

### Rules of thumb for hardware

- **8 GB RAM:** E2B, slow but works.
- **16 GB RAM / Apple Silicon:** E4B or 12B comfortably.
- **24 GB GPU or 32 GB+ unified memory:** 26B MoE, 31B or Qwen3.6 27B.
- **No usable machine?** Ask your Fest host what compute is available. Ollama also offers hosted `-cloud` tags.

## Step 2: Run the model on your machine

Ollama is the quickest path: one install, one command per model, and a local API on port 11434. It is free, open source, and every model in its library is open-weight.

1. Install Ollama. On macOS or Windows, download it from [ollama.com/download](https://ollama.com/download). On Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. Pull the chat model and an embedding model (the embedding model turns text into vectors for search):

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

3. Say hello from the terminal to confirm it works:

```bash
ollama run gemma4:e4b "Explain what a city council consent agenda is in two sentences."
```

4. Set up the Python project:

```bash
mkdir civic-qa && cd civic-qa
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install ollama pypdf numpy gradio pandas
```

Prefer Hugging Face? The same models are published there and run with the `transformers` library. Ollama is used here because it handles quantization and chat templates for you, which saves real time on a Hack Day.

## Step 3: Build "Ask Your City Council"

The tool uses retrieval-augmented generation (RAG): instead of trusting what the model remembers, it searches the actual records and makes the model answer only from what it finds. That is what makes page citations possible.

Using minutes from [City of Oakwood](https://oakwoodohio.gov/city-council-meetings/)

* Note: The PDF's downloaded from cite are not machine readable. You would have to use OCR to read these in first!  

How it works:

1. **Ingest (once):** split each PDF into page-tagged chunks, turn each chunk into a vector, save the index.
2. **Retrieve (per question):** embed the question, find the 5 most similar chunks.
3. **Answer:** send those chunks plus strict instructions to the local model; it replies in plain language with `[file, p. N]` citations.

### 3a. Get the documents

Download 3 to 10 recent meeting minutes or agendas as PDFs from your city or county website (most publish them on a "Meetings" or "Agendas & Minutes" page). Put them in a `docs/` folder. Text-based PDFs work best; scanned images need OCR first (see Step 4 ideas).

Download 3 to 10 recent meeting minutes or agendas as PDFs from your city or county website (most publish them on a "Meetings" or "Agendas & Minutes" page). Put them in a `docs/` folder. This tutorial uses minutes from the \[City of Oakwood, Ohio\](https://oakwoodohio.gov/city-council-meetings/). Like many small cities, Oakwood posts minutes scanned on an office copier, so they have no text a computer can read. Run the OCR step below before ingesting.

### OCR: make scanned minutes searchable

Optical character recognition (OCR) turns a picture of a page into text. Without it, `ingest.py` finds nothing in a scanned PDF and the tool answers every question with "I couldn't find that." On Oakwood's May 4, 2026 minutes, the extracted text went from 0 characters on all 9 pages to 1,855–4,603 characters per page, in about 17 seconds.

**Check whether a PDF needs OCR.** If this prints 0 for every page, the PDF is a scan:

```python
from pypdf import PdfReader
for n, page in enumerate(PdfReader("docs/may-4-2026-council-minutes.pdf").pages, 1):
    print(n, len(page.extract_text() or ""))
```

**Install OCRmyPDF,** an open-source tool built on the Tesseract OCR engine. It adds an invisible text layer to each page and keeps the page order, so page citations still work.

- **macOS:** `brew install ocrmypdf`
- **Ubuntu/Debian:** `sudo apt install ocrmypdf`
- **Windows:** easiest under WSL with the Ubuntu command; otherwise install Tesseract and Ghostscript first.

Then add the Python package to your project: `pip install ocrmypdf`.

**`ocr.py`**

```python
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
```

Run `python ocr.py`, then build the index from the OCR'd copies: `python ingest.py docs_ocr`. Keep the originals in `docs/` so residents can always be pointed to the official record.

**Check the OCR before you trust it.** On the Oakwood minutes, most prose came out clean, but:

- Roll-call lines with dot leaders ("MRS. LEIGH TURBEN ........ YEA") turned into strings of junk letters, and a council member's name was misread there.
- "Pickleball" became "picklebal\]" in one sentence.
- The minutes book's printed page numbers (386–394) differ from the PDF's page numbers (1–9). Citations use PDF pages; say so in your UI.

Names, vote counts and dollar amounts are exactly what residents ask about, so spot-check them against the scanned image.

**Tougher scans** (handwriting, stamps, tables, faint copies): an open vision model can read the page image directly. Gemma 4 accepts images through Ollama, and its model card recommends a high image token budget for OCR and small text. Ai2's olmOCR is a fully open model built for document OCR. Both are slower than Tesseract, so use them on the pages `ocr.py` flags rather than on everything.


### 3b. `ingest.py`

`ingest.py` reads PDFs plus .txt and .md files (useful for agendas, transcripts or notes you've already cleaned up). PDF chunks are tagged by page; text chunks by line range, and Markdown files split at each heading so sections stay together.

```python
"""Build a search index from a folder of public-meeting PDFs."""
import json
import pathlib
import sys

import numpy as np
import ollama
from pypdf import PdfReader

EMBED_MODEL = "nomic-embed-text"
CHUNK_CHARS = 1200   # roughly 250 words per chunk
BATCH = 32


def chunks_from_pdf(path):
    """Yield page-tagged text chunks so every answer can cite a page."""
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        for i in range(0, len(text), CHUNK_CHARS):
            piece = text[i:i + CHUNK_CHARS]
            if len(piece) > 50:
                yield {"source": path.name, "page": page_num, "text": piece}


def main(folder):
    pdfs = sorted(pathlib.Path(folder).glob("*.pdf"))
    chunks = [c for pdf in pdfs for c in chunks_from_pdf(pdf)]
    vectors = []
    for i in range(0, len(chunks), BATCH):
        batch = [f"search_document: {c['text']}" for c in chunks[i:i + BATCH]]
        vectors.extend(ollama.embed(model=EMBED_MODEL, input=batch)["embeddings"])
        print(f"Embedded {min(i + BATCH, len(chunks))}/{len(chunks)}")
    np.save("index.npy", np.array(vectors, dtype=np.float32))
    pathlib.Path("chunks.json").write_text(json.dumps(chunks))
    print(f"Indexed {len(chunks)} chunks from {len(pdfs)} PDFs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs")
```

Run it with `python ingest.py docs`. The `search_document:` / `search_query:` prefixes are what nomic-embed-text expects and noticeably improve search quality.

### 3c. `ask.py`

The system prompt does most of the civic work: it forbids outside knowledge, sets a reading level, requires citations and tells the model to admit when it doesn't know.

```python
"""Answer questions about the indexed documents, with page citations."""
import json
import pathlib
import sys

import numpy as np
import ollama

CHAT_MODEL = "gemma4:e4b"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 5

SYSTEM = """You help residents understand their local government.
Use ONLY the excerpts provided. Do not use outside knowledge.
Write in plain language at about an 8th-grade reading level.
After each fact, cite its excerpt like [minutes-2026-09-15.pdf, p. 4].
If the excerpts do not answer the question, say:
"I couldn't find that in these records." Then suggest contacting the city clerk.
Never guess names, dates, dollar amounts, or vote counts."""

VECTORS = np.load("index.npy")
VECTORS /= np.linalg.norm(VECTORS, axis=1, keepdims=True)
CHUNKS = json.loads(pathlib.Path("chunks.json").read_text())


def retrieve(question, k=TOP_K):
    q = ollama.embed(model=EMBED_MODEL, input=f"search_query: {question}")["embeddings"][0]
    q = np.array(q, dtype=np.float32)
    scores = VECTORS @ (q / np.linalg.norm(q))
    return [CHUNKS[i] for i in np.argsort(scores)[::-1][:k]]


def answer(question):
    hits = retrieve(question)
    context = "\n\n".join(
        f"Excerpt [{h['source']}, p. {h['page']}]:\n{h['text']}" for h in hits
    )
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]


if __name__ == "__main__":
    print(answer(" ".join(sys.argv[1:]) or "What did the council vote on last month?"))
```

Try it: `python ask.py "What did the council decide about road repairs?"`. Low temperature (0.2) keeps answers factual rather than creative.

### 3d. `app.py`: a web page residents can use

```python
"""A simple web UI for residents."""
import gradio as gr

from ask import answer

demo = gr.ChatInterface(
    fn=lambda message, history: answer(message),
    title="Ask Your City Council",
    description="Answers come only from published meeting records, with page citations. "
                "Always check the original document before relying on an answer.",
    examples=["What did the council decide about the new bike lanes?",
              "How much is the city spending on road repairs this year?"],
)

if __name__ == "__main__":
    demo.launch()
```

Run `python app.py` and open the local URL it prints. That's a demo-ready Hack Day project.

### 3e. Test it before you demo

Write five questions you know the answers to, plus two the documents can't answer. Check that every answer cites a real page, and that the unanswerable ones get "I couldn't find that." If the model invents facts, lower `TOP_K`, tighten the prompt, or move up to `gemma4:12b`.

### 3f. Add-on: has 311 demand grown enough to justify a budget renewal?

This add-on answers a budget-hearing question: did requests for a service go up, and what does the city spend per request? Pandas does all the arithmetic; the model only explains the finished table, because language models are unreliable at math.

**Inputs**

- **311 requests:** most cities publish these as a CSV on their open data portal. Look for a date column (e.g. `created_date`) and a type column (e.g. `request_type`) and set them at the top of the script.
- **Budget:** build a small `budget.csv` by hand from the adopted budget PDFs (or use the Step 3 tool to find the pages). Use the same category names as the 311 data:

```csv
year,category,budget_usd
2024,Pothole,500000
2025,Pothole,520000
2024,Streetlight,210000
```

**`trends.py`**

```python
"""Do 311 requests justify renewing a program's budget?

Counts 311 requests per year and category, joins them to the budget,
and computes year-over-year change and cost per request.
The math happens here in pandas; the model only explains the results.
"""
import sys

import pandas as pd

# Column names vary by city. Check your CSV header and edit these.
DATE_COL = "created_date"
CATEGORY_COL = "request_type"


def requests_by_year(csv_311, categories=None):
    """Return a table of request counts per year and category."""
    df = pd.read_csv(csv_311, usecols=[DATE_COL, CATEGORY_COL])
    df["year"] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    if categories:
        df = df[df[CATEGORY_COL].isin(categories)]
    return (df.groupby(["year", CATEGORY_COL]).size()
              .rename("requests").reset_index()
              .rename(columns={CATEGORY_COL: "category"}))


def demand_vs_budget(csv_311, csv_budget, categories=None):
    """Join yearly 311 demand to budget lines.

    csv_budget needs three columns: year, category, budget_usd
    (one row per program per year, typed in from the adopted budget PDFs).
    """
    counts = requests_by_year(csv_311, categories)
    budget = pd.read_csv(csv_budget)
    out = counts.merge(budget, on=["year", "category"], how="left")
    out = out.sort_values(["category", "year"])
    g = out.groupby("category")
    out["requests_change_pct"] = (g["requests"].pct_change() * 100).round(1)
    out["budget_change_pct"] = (g["budget_usd"].pct_change() * 100).round(1)
    out["usd_per_request"] = (out["budget_usd"] / out["requests"]).round(2)
    return out


def summarize(table, chat_model="gemma4:e4b"):
    """Ask the local model to explain the computed table in plain language."""
    import ollama
    prompt = (
        "Here is a table of 311 service requests and budget by year:\n\n"
        f"{table.to_string(index=False)}\n\n"
        "In plain language for a budget hearing, say whether demand rose, fell or held "
        "steady for each category, quoting only numbers that appear in the table. "
        "Note years with missing data. Do not calculate new numbers. "
        "Do not claim the budget caused or should change because of these trends; "
        "just describe them."
    )
    response = ollama.chat(model=chat_model,
                           messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.2})
    return response["message"]["content"]


if __name__ == "__main__":
    table = demand_vs_budget(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
    table.to_csv("demand_vs_budget.csv", index=False)
    print(table.to_string(index=False))
    if "--no-ai" not in sys.argv:
        print("\n" + summarize(table))
```

Run it after `pip install pandas`:

```bash
python trends.py 311_requests.csv budget.csv Pothole Streetlight
```

On a test file, the output looked like this; the model then turns it into a short paragraph:

| year | category | requests | budget\_usd | requests\_change\_pct | budget\_change\_pct | usd\_per\_request |
| --- | --- | --- | --- | --- | --- | --- |
| 2024 | Pothole | 476 | 500,000 | 24.3 | 0.0 | 1,050.42 |
| 2025 | Pothole | 639 | 520,000 | 34.2 | 4.0 | 813.77 |

Read the result with care before it goes into a budget request:

- **Compare full years.** If the current year is partial, compare the same months (e.g. Jan–Sep) across years, or the latest year will look like a drop.
- **More requests isn't always more need.** A new 311 app, a publicity campaign or a category being renamed can raise counts without any change on the street. Check for those before claiming growth.
- **Falling cost per request** (as in the sample: $1,050 to $814) can support a renewal as an efficiency story, or signal the program is stretched thin. Say which, with evidence from staff.
- Open `demand_vs_budget.csv` in a spreadsheet to chart it for the hearing.

## Worked example: asking Dayton's data a civic question

In a few minutes, AI turned a raw city CSV into a clear answer: welfare checks are Dayton residents' top reason for calling police. The same analysis also showed what the data could not answer, which mattered just as much.

**The question:** "What are the trends for potholes, trash and other complaints, and what is the top complaint?"

**The data:** Dayton's [Calls for Service dataset](https://dayton-transparency-portal-1-daytonohio.hub.arcgis.com/datasets/callsforservice-opendata-hosted) from the city's transparency portal. The download covered 25,824 police calls from Aug 6 to Oct 4, 2026, with call type, date, neighborhood and priority board.

### How the AI helped

1. **It read the file before answering.** The header showed these were police dispatches, not 311 service requests, so there was no pothole category at all. Trash appears only as "Dumping or Littering."
2. **Code did the counting.** The AI wrote a short pandas script and ran it, so every number comes from the data rather than the model's memory.
3. **It chose fair comparisons.** It counted only calls from residents (leaving out 4,775 that officers started themselves) and compared two complete four-week windows, so partial weeks wouldn't fake a trend.
4. **It explained the results in plain language,** with caveats a council member would need.

```python
"""What are Dayton residents calling about, and what's changing?"""
import pandas as pd

df = pd.read_csv("CallsForService.csv", encoding="utf-8-sig")
df["when"] = pd.to_datetime(df["Call_Date"], format="%m/%d/%Y %I:%M:%S %p")

# Keep calls from residents; drop ones officers started themselves.
residents = df[df["Self_Initiated"].str.strip() == "no"]

# Top complaints overall
print(residents["Dispatch_Type"].value_counts().head(10))

# Compare two complete four-week windows so partial weeks don't skew trends
before = residents[(residents.when >= "2026-08-09") & (residents.when < "2026-09-06")]
after = residents[(residents.when >= "2026-09-06") & (residents.when < "2026-10-04")]
trend = pd.DataFrame({
    "aug_9_to_sep_5": before["Dispatch_Type"].value_counts(),
    "sep_6_to_oct_3": after["Dispatch_Type"].value_counts(),
}).fillna(0).astype(int)
trend["change_pct"] = ((trend.iloc[:, 1] - trend.iloc[:, 0]) / trend.iloc[:, 0] * 100).round(0)
watch = ["WELFARE CHECK", "MISSING", "ALARM COMPLAINT", "PARKING COMPLAINT",
         "LOUD NOISE COMPLAINT ETC", "ABANDONED VEHICLE", "DUMPING OR LITTERING", "TREE DOWN"]
print(trend.loc[watch])

# Is there a pothole category at all?
print(df["Dispatch_Type"].str.contains("POTHOLE|POT HOLE").any())
```

### What it found

**Top complaint:** welfare checks, with 1,747 resident calls (about 8%), ahead of accidents (1,254) and disorderly subjects (1,122). Total resident calls held flat across the two windows (10,088 vs. 9,883).

| Call type | Aug 9–Sep 5 | Sep 6–Oct 3 | Change |
| --- | --- | --- | --- |
| Welfare check | 811 | 862 | +6% |
| Missing person | 309 | 404 | +31% |
| Alarm complaint | 443 | 381 | −14% |
| Parking complaint | 212 | 218 | +3% |
| Loud noise | 180 | 197 | +9% |
| Abandoned vehicle | 90 | 95 | +6% |
| Dumping or littering | 21 | 18 | −14% |
| Tree down | 40 | 3 | −92% |

### What the data can't tell you

- **Potholes:** not in this dataset. Residents report them to the city, not to police.
- **Trash:** about 5 dumping calls a week is too few to call a trend. Most dumping calls that list a board came from Southeast (14 of 26), but Southeast also has the most calls of every kind.
- **Budget renewal:** eight weeks can't show year-over-year growth. Use multi-year service-request data with the `trends.py` script from 3f.
- **Big swings need a reason.** The tree-down drop may reflect a single August storm rather than a lasting change. Check the weather or news before reporting a spike.

### Where to get Dayton service-request (311-style) data

Dayton doesn't use "311" as a name. Residents report potholes, illegal dumping, missed bulk trash and similar issues through **Dayton Delivers** (the city's request app) or the "Request Service / Report Problem" link on the city's [I Want To… page](https://daytonohio.gov/9). They can also call 937-333-3333, or 937-333-4800 for Public Works and waste collection.

For the data itself:

- Start at the [Dayton Transparency Portal](https://dayton-transparency-portal-1-daytonohio.hub.arcgis.com/), which hosts the calls-for-service data above, and search for service-request datasets.
- Budget and spending data lives on Dayton's [OpenGov transparency site](https://daytonoh.opengov.com/transparency).
- If Dayton Delivers requests aren't published as a download, ask for them through the city's public records request process. Request two or more years with request type and date.

## Step 4: Extend it, or pick another civic project

Each idea below reuses the Step 3 pattern (local model + public data + citations) and fits in one Hack Day.

| Project | Who it helps | Open-model angle |
| --- | --- | --- |
| Meeting-audio summarizer | Residents who can't attend evening meetings | Gemma 4 E2B/E4B accept audio input; transcribe, then summarize with timestamps |
| Scanned-records reader | Clerks and researchers with old paper archives | Use Gemma 4's vision input at a high image token budget, or Ai2's open olmOCR model, to turn scans into searchable text |
| Multilingual notices | Non-English-speaking households | Translate council summaries and service notices; Qwen is strong on multilingual tasks. Have a native speaker review |
| 311 request router | City staff triaging service requests | Classify free-text complaints (pothole, streetlight, trash) into department categories, keeping resident data on city machines |
| Benefits plain-language explainer | People applying for food, housing or utility aid | RAG over official program rules; answers cite the rule and link to the real application |
| Budget explorer | Journalists, advocates, curious residents | Let people ask "how much went to parks?" over the published budget PDF, with page citations |
| Alt-text helper for public sites | Blind and low-vision residents | Draft image descriptions for government web pages and PDFs for a human to approve |

Two quick upgrades to the Step 3 tool: add a "Show sources" panel that prints the retrieved excerpts, and add a nightly script that downloads newly posted minutes and re-runs `ingest.py`.

## Responsible civic AI checklist

A wrong answer about a tax deadline or a benefit rule can cost someone real money, so civic tools need a higher bar than a typical demo. Check each item before you demo or share.

- [ ] **Cite or stay silent.** Every factual answer links to a document and page; the tool says "I couldn't find that" rather than guessing.
- [ ] **Label it clearly.** The page says it's an AI tool, not an official city service, and tells people to verify with the original record.
- [ ] **Keep personal data local.** No resident names, addresses or case details sent to outside APIs. Running locally makes this the default; don't undo it with logging.
- [ ] **Use only public records.** Confirm the documents are already published. Don't ingest anything obtained through a records request without checking what can be shared.
- [ ] **Write plainly.** Short sentences, defined jargon ("consent agenda", "first reading"), and an 8th-grade reading level target.
- [ ] **Make it accessible.** Keyboard-navigable UI, sufficient contrast, screen-reader labels. Gradio's defaults are a start, not the finish.
- [ ] **Test for uneven quality.** Try questions about different neighborhoods and in different languages; note where answers get worse.
- [ ] **Have a human in the loop for translations and alt text.** Machine output is a draft.
- [ ] **Check the licenses.** Model license (Apache 2.0 for the models used here), data terms, and your own project's license.
- [ ] **Document limits in the README.** Which documents are covered, date range, known failure cases, and the model used.

## Step 5: Ship it and share it

PRs no longer earn Hacktoberfest swag, so this year the goal is a project worth showing and contributions maintainers actually want.

**At a Hack Day**

1. Push your project to a public GitHub repo with an open-source license (MIT or Apache 2.0) and a README covering setup, the model used and known limits.
2. Record a 2-minute demo: one good answer with a citation, one honest "I couldn't find that."
3. Submit through the Challenges tab on your Fest's event page before the host's deadline.

**Online**

- Write up what you built as a DEV post and enter the [Hacktoberfest DEV Challenges](https://dev.to/challenges); connect your DEV account in My Hacktoberfest for the badge.

**Contributing to existing civic projects**

- Browse the [civic-tech topic on GitHub](https://github.com/topics/civic-tech) and look for issues labelled `good first issue` or `help wanted`.
- Read CONTRIBUTING.md first and comment on the issue before starting.
- Only open a PR you have run and tested yourself. If AI helped write it, say so in the description. Maintainers' complaints about low-effort PRs are why the rules changed.

## Sources

- [Hacktoberfest 2026 FAQ](https://hacktoberfest.com/questions/)
- [Find a Fest](https://hacktoberfest.com/fests/) · [My Hacktoberfest](https://hacktoberfest.com/my/) · [DEV Challenges](https://dev.to/challenges)
- [Ollama: gemma4 model page](https://ollama.com/library/gemma4) (sizes, context windows, audio support)
- [Ollama: qwen3.6 model page](https://ollama.com/library/qwen3.6)
- [Ars Technica: Gemma 4 switches to Apache 2.0](https://arstechnica.com/ai/2026/04/google-announces-gemma-4-open-ai-models-switches-to-apache-2-0-license/)
- [Ai2: Olmo 3 announcement](https://secure.businesswire.com/news/home/20251120271934/en/Olmo-3-Charting-a-Path-Through-the-Model-Flow-to-Lead-Open-Source-AI)
- [Ollama download](https://ollama.com/download)

* [Dayton Calls for Service dataset](https://dayton-transparency-portal-1-daytonohio.hub.arcgis.com/datasets/callsforservice-opendata-hosted)
* [City of Dayton: I Want To…](https://daytonohio.gov/9) (service requests, phone numbers)
* [Dayton Daily News: Dayton Delivers app](https://daytondailynews.com/news/new-dayton-app-helps-city-deliver-services/k9mx2vkmwQdVoGfrMKUxdI)
* [NLC: Dayton's transparency portal](https://www.nlc.org/article/2018/10/25/dayton-sets-an-example-on-transparency/)

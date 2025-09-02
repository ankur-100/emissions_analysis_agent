# Emissions Analysis & Insights Agent

Prototype for Unravel Carbon — Round 2 Take‑Home.

## Quick Start

```bash
# 1) Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install -r requirements.txt

# 3) Set your key (Gemini free tier)
export GOOGLE_API_KEY=YOUR_KEY   # (Windows PowerShell: $env:GOOGLE_API_KEY="YOUR_KEY")

# 4) Run the app
streamlit run app.py
```

Upload the provided CSVs and PDFs in the left sidebar (or place them in `data/` named `scope1.csv`, `scope2.csv`, `scope3.csv`, and the peer / GHG PDFs).

## What’s inside?

- `app.py` — Streamlit UI with 4 tabs: **Overview**, **Ask Your Data**, **Peer Compare**, **Quality Assessment**, **Report**.
- `src/rag_utils.py` — PDF ingestion (chunking), vector store (FAISS), Gemini RAG pipeline.
- `src/quality.py` — Rule‑based quality checks & scoring aligned with GHG Protocol patterns.
- `src/prompts.py` — Structured prompts & few‑shots for education, SQL, and report writing.
- `src/peers.py` — Lightweight peer‑report parsing via LLM extraction helpers (no heavy PDF table deps).
- `requirements.txt` — Dependencies.
- `overview.md` — One‑page write‑up for the deliverable.

## Notes

- Uses **Gemini 2.5 Flash** for reasoning + **text-embedding-004** for retrieval.
- No heavy PDF table libs (Camelot/Tabula); instead, we do text extraction and LLM‑guided field extraction for peer totals.
- Natural language → **Pandas/SQL hybrid**: the agent chooses between:
  1) structured Pandas analysis tools; or
  2) report/education answers via RAG over GHG Protocol PDF.

## Evaluation Ideas

- Small golden‑set Q&A with expected answers.
- Retrieval quality with **RAGAS** (optional) or simple precision@k.
- LLM‑as‑judge rubric for report quality (clarity, correctness, actionability).
- Unit tests for quality rules and schema detection (pytest).


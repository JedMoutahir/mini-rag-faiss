# Mini-RAG with FAISS (CPU)

A tiny, production-style **RAG** system:
- **PDF ingestion** → text extraction (PyPDF2), chunking with overlap
- **Embeddings** → TF-IDF (scikit-learn) → L2-normalized dense vectors
- **Index** → FAISS (IndexFlatIP) for fast cosine similarity
- **Query** → top-k retrieval + simple extractive summarization
- **Fully offline** (no model downloads), unit-tested, CPU-only by default

> Designed to scale: you can throw many PDFs at it; it streams pages,
> builds incremental TF-IDF vocabulary, and writes FAISS + metadata to disk.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Ingest PDFs from a folder and build an index
python ingest.py --pdf_dir ./docs --out_dir ./store --chunk_size 800 --chunk_overlap 150

# 2) Ask questions
python query.py --store ./store --k 5 --question "What is the warranty period?"

# 3) (Optional) Print sources
python query.py --store ./store --k 5 --question "..." --show_sources
```

## Files
rag/chunk.py : page-aware chunking with overlap and sentence boundaries

rag/index.py : fit TF-IDF, make dense vectors, write/read FAISS + metadata

rag/search.py : retrieval (cosine via FAISS), MMR reranking

rag/summarize.py : simple extractive summarizer

ingest.py : CLI to build the store

query.py : CLI to query the store

## Store artifacts
```bash
store/
├── faiss.index            # FAISS IndexFlatIP
├── tfidf.pkl              # fitted TfidfVectorizer (pickle)
├── meta.jsonl             # metadata per chunk (doc id, page, offsets)
└── docs.jsonl             # docs catalog (path, page_count, id)
```

## Scale up
Use more/larger PDFs: ingestion streams pages; TF-IDF grows vocab.

Use GPU-FAISS: swap to faiss.index_cpu_to_all_gpus(index) (see comments).

Replace TF-IDF with SOTA embeddings (e.g., sentence-transformers)
by plugging into rag/index.py:embed_texts.

## Testing
```bash
pytest -q
```
Tests generate small PDFs programmatically (no network).
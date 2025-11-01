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

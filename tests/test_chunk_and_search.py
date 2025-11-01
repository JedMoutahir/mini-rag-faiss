import os, json, tempfile, shutil
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from rag.index import ingest_pdfs, load_store, embed_texts
from rag.search import faiss_search

def make_pdf(path: str, lines):
    c = canvas.Canvas(path, pagesize=letter)
    w, h = letter
    y = h - 72
    for line in lines:
        c.drawString(72, y, line)
        y -= 14
        if y < 72:
            c.showPage()
            y = h - 72
    c.save()

def test_ingest_and_search(tmp_path):
    docs_dir = tmp_path / "docs"
    store = tmp_path / "store"
    docs_dir.mkdir()
    make_pdf(docs_dir / "a.pdf", [
        "Paris is the capital of France. The Eiffel Tower is in Paris.",
        "France is in Europe. Capital cities often have landmarks.",
    ])
    make_pdf(docs_dir / "b.pdf", [
        "Berlin is the capital of Germany.",
        "Germany is also in Europe."
    ])

    ingest_pdfs(str(docs_dir), str(store), chunk_size=120, chunk_overlap=30)
    # ensure artifacts
    assert (store / "faiss.index").exists()
    assert (store / "tfidf.pkl").exists()
    assert (store / "meta.jsonl").exists()
    assert (store / "docs.jsonl").exists()
    assert (store / "chunks.jsonl").exists()

    index, tfidf, metas, docs = load_store(str(store))
    q = "What is the capital of France?"
    qv = embed_texts(tfidf, [q])
    D, I = faiss_search(index, qv, topk=3)
    assert I.shape[1] == 3

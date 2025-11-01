from __future__ import annotations
import os, pickle, json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
from .chunk import chunk_sentences
from .utils import ensure_dir, write_jsonl, read_jsonl, sha1

@dataclass
class StorePaths:
    out_dir: str
    faiss_path: str
    tfidf_path: str
    meta_path: str
    docs_path: str

def build_paths(out_dir: str) -> StorePaths:
    ensure_dir(out_dir)
    return StorePaths(
        out_dir=out_dir,
        faiss_path=os.path.join(out_dir, "faiss.index"),
        tfidf_path=os.path.join(out_dir, "tfidf.pkl"),
        meta_path=os.path.join(out_dir, "meta.jsonl"),
        docs_path=os.path.join(out_dir, "docs.jsonl"),
    )

def read_pdf_texts(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt)
    return pages

def ingest_pdfs(pdf_dir: str, out_dir: str, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
    paths = build_paths(out_dir)
    docs = []
    metas = []
    texts = []

    # Collect PDFs
    pdfs = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    pdfs.sort()
    doc_id = 0
    for pdf in pdfs:
        pages = read_pdf_texts(pdf)
        docs.append({"id": doc_id, "path": os.path.abspath(pdf), "page_count": len(pages)})
        for page_idx, page_txt in enumerate(pages):
            for (s, e, chunk) in chunk_sentences(page_txt, chunk_size, chunk_overlap):
                metas.append({"doc_id": doc_id, "page": page_idx, "start": s, "end": e})
                texts.append(chunk)
        doc_id += 1

    if not texts:
        # still write empty store for consistency
        write_jsonl(paths.docs_path, docs)
        write_jsonl(paths.meta_path, metas)
        # empty tfidf/index
        with open(paths.tfidf_path, "wb") as f:
            pickle.dump(TfidfVectorizer(), f)
        index = faiss.IndexFlatIP(1)
        faiss.write_index(index, paths.faiss_path)
        return

    # Fit TF-IDF and embed
    tfidf = TfidfVectorizer(max_features=200000, lowercase=True, stop_words="english")
    X = tfidf.fit_transform(texts)
    # L2 normalize to make inner product == cosine
    X = X.astype(np.float32)
    norms = np.sqrt((X.power(2)).sum(axis=1)).A1 + 1e-12
    X = X.multiply(1.0 / norms[:, None]).astype(np.float32)

    # Convert to dense (WARNING: memory heavy for very large corpora; ok for demos/small-medium)
    dense = X.toarray().astype(np.float32)

    # Build FAISS (cosine via IP since vectors are normalized)
    index = faiss.IndexFlatIP(dense.shape[1])
    index.add(dense)

    # Save artifacts
    with open(paths.tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    faiss.write_index(index, paths.faiss_path)
    write_jsonl(paths.docs_path, docs)
    write_jsonl(paths.meta_path, metas)

def load_store(out_dir: str):
    paths = build_paths(out_dir)
    index = faiss.read_index(paths.faiss_path)
    with open(paths.tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    metas = list(read_jsonl(paths.meta_path))
    docs = list(read_jsonl(paths.docs_path))
    return index, tfidf, metas, docs

def embed_texts(tfidf: TfidfVectorizer, texts: List[str]) -> np.ndarray:
    X = tfidf.transform(texts).astype(np.float32)
    norms = np.sqrt((X.power(2)).sum(axis=1)).A1 + 1e-12
    X = X.multiply(1.0 / norms[:, None]).astype(np.float32)
    return X.toarray().astype(np.float32)

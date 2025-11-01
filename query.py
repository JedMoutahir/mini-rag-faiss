import argparse, json
import numpy as np
from rag.index import load_store, embed_texts
from rag.search import faiss_search, mmr
from rag.summarize import extractive_summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=str, required=True)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lambda_mmr", type=float, default=0.5)
    ap.add_argument("--show_sources", action="store_true")
    args = ap.parse_args()

    index, tfidf, metas, docs = load_store(args.store)
    q_vec = embed_texts(tfidf, [args.question])
    D, I = faiss_search(index, q_vec, topk=max(args.k*3, args.k))
    idxs = I[0]

    # load all chunk texts by re-embedding to get dense vectors (already normalized)
    # NOTE: to reconstruct chunk text, we need the tf-idf vocabulary; we don't store raw text.
    # Instead, we will store raw chunk text in future; for now, we rebuild from meta is not possible.
    # -> Adjust: keep a shadow 'chunks.jsonl' file with raw text (add in index.py).
    # Quick fix: read chunks.jsonl
    import os, json
    chunks_path = os.path.join(args.store, "chunks.jsonl")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(l) for l in f if l.strip()]
    texts = [chunks[i]["text"] for i in range(len(chunks))]

    # gather candidate vectors for MMR
    from rag.index import embed_texts as embed
    cand_texts = [texts[i] for i in idxs]
    cand_vecs = embed(tfidf, cand_texts)
    selected_rel = mmr(q_vec[0], cand_vecs, k=args.k, lambda_=args.lambda_mmr)
    final_idx = [idxs[i] for i in selected_rel]

    top_texts = [texts[i] for i in final_idx]
    answer = extractive_summary(args.question, top_texts, max_sentences=6)
    print(answer)

    if args.show_sources:
        print("\n--- Sources ---")
        for i in final_idx:
            m = metas[i]
            d = [d for d in docs if d["id"] == m["doc_id"]][0]
            print(f"- {d['path']} (page {m['page']+1})")

if __name__ == "__main__":
    main()

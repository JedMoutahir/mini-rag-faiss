from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

def faiss_search(index: faiss.Index, query_vecs: np.ndarray, topk: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(query_vecs.astype(np.float32), topk)
    return D, I

def mmr(query: np.ndarray, candidates: np.ndarray, k: int, lambda_: float = 0.5):
    """
    Maximal Marginal Relevance on cosine-sim normalized vectors.
    query: (d,), candidates: (n, d)
    returns indices
    """
    selected = []
    sim_to_query = candidates @ query
    sim_matrix = candidates @ candidates.T
    while len(selected) < min(k, len(candidates)):
        if not selected:
            idx = int(np.argmax(sim_to_query))
            selected.append(idx)
            continue
        rest = [i for i in range(len(candidates)) if i not in selected]
        mmr_scores = []
        for i in rest:
            div = max(sim_matrix[i, selected]) if selected else 0.0
            mmr_scores.append(lambda_ * sim_to_query[i] - (1 - lambda_) * div)
        idx = rest[int(np.argmax(mmr_scores))]
        selected.append(idx)
    return selected

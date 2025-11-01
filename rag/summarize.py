from __future__ import annotations
import re
from typing import List

def extractive_summary(question: str, chunks: List[str], max_sentences: int = 5) -> str:
    """
    Very simple extractive summary:
    - rank sentences by presence of query keywords (case-insensitive)
    - return top-k unique sentences preserving order
    """
    q_tokens = set(t for t in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(t) > 1)
    scored = []
    for i, ch in enumerate(chunks):
        sents = re.split(r"(?<=[.!?])\s+", ch.strip())
        for j, s in enumerate(sents):
            toks = set(re.findall(r"[A-Za-z0-9]+", s.lower()))
            score = len(q_tokens & toks)
            if score > 0:
                scored.append((i, j, score, s))
    # sort by score desc, then keep order by (i,j)
    scored.sort(key=lambda x: (-x[2], x[0], x[1]))
    out = []
    seen = set()
    for _, _, _, s in scored:
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_sentences:
            break
    return " ".join(out) if out else " ".join(chunks[:1])  # fallback first chunk

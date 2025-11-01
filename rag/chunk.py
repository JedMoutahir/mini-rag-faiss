from __future__ import annotations
import re
from typing import List, Tuple

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])")

def sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]

def chunk_sentences(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Tuple[int, int, str]]:
    """
    Return chunks as (start_char, end_char, chunk_text) from text.
    Uses sentence packing to fit ~chunk_size chars with overlap.
    """
    sents = sentences(text)
    if not sents:
        return []
    chunks = []
    i = 0
    positions = []
    cursor = 0
    # map sentence to [start,end)
    for s in sents:
        start = text.find(s, cursor)
        if start == -1:
            start = cursor
        end = start + len(s)
        positions.append((start, end, s))
        cursor = end

    while i < len(positions):
        cur = positions[i]
        cur_len = len(cur[2])
        start = positions[i][0]
        buf = [cur[2]]
        j = i + 1
        total = cur_len
        while j < len(positions) and total + 1 + len(positions[j][2]) <= chunk_size:
            buf.append(positions[j][2])
            total += 1 + len(positions[j][2])
            j += 1
        end = positions[j-1][1]
        chunks.append((start, end, " ".join(buf)))
        # move forward with overlap in characters
        next_start_pos = max(end - chunk_overlap, start + 1)
        # find sentence index that starts after next_start_pos
        k = j
        while k > i and positions[k-1][0] > next_start_pos:
            k -= 1
        # advance by at least one sentence
        i = max(k, i + 1)
    return chunks

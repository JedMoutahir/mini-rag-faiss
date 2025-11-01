import os
import subprocess
import sys
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

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

def test_cli_end_to_end(tmp_path):
    docs_dir = tmp_path / "docs"
    store = tmp_path / "store"
    docs_dir.mkdir()
    make_pdf(docs_dir / "guide.pdf", [
        "Warranty: The product includes a two-year warranty from the date of purchase.",
        "Support is available online.",
    ])

    # ingest
    proc = subprocess.run(
        [sys.executable, "ingest.py", "--pdf_dir", str(docs_dir), "--out_dir", str(store), "--chunk_size", "160", "--chunk_overlap", "40"],
        check=True, capture_output=True, text=True
    )
    assert "Store written to" in proc.stdout

    # query
    proc2 = subprocess.run(
        [sys.executable, "query.py", "--store", str(store), "--question", "What is the warranty period?", "--k", "3", "--show_sources"],
        check=True, capture_output=True, text=True
    )
    out = proc2.stdout.lower()
    # should mention "two-year" (allow "two year")
    assert "two" in out and "year" in out

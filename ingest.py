import argparse
from rag.index import ingest_pdfs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--chunk_size", type=int, default=800)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    args = ap.parse_args()
    ingest_pdfs(args.pdf_dir, args.out_dir, args.chunk_size, args.chunk_overlap)
    print(f"Store written to: {args.out_dir}")

if __name__ == "__main__":
    main()

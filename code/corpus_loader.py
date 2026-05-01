"""Load and chunk support corpus documents from the data/ directory."""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    text: str
    source: str           # file path relative to data/
    company: str          # HackerRank, Claude, or Visa
    product_area: str     # directory/category name
    chunk_index: int


def clean_markdown(text: str) -> str:
    """Strip excessive markdown formatting for cleaner chunks."""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*\*|__', '', text)
    text = re.sub(r'`{1,3}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def sliding_window_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                          overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def load_markdown_file(path: Path) -> str:
    """Read a markdown file, return cleaned text."""
    try:
        text = path.read_text(encoding='utf-8')
        return clean_markdown(text)
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return ""


def discover_corpus() -> List[Chunk]:
    """
    Walk data/ and return all chunked documents.
    Structure:
      data/hackerrank/.../*.md
      data/claude/.../*.md
      data/visa/.../*.md
    """
    chunks: List[Chunk] = []
    company_map = {
        "hackerrank": "HackerRank",
        "claude": "Claude",
        "visa": "Visa",
    }

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    for company_dir in DATA_DIR.iterdir():
        if not company_dir.is_dir():
            continue
        company_raw = company_dir.name.lower()
        company = company_map.get(company_raw)
        if not company:
            continue

        md_files = list(company_dir.rglob("*.md"))
        for md_file in md_files:
            rel = md_file.relative_to(DATA_DIR)
            product_area = md_file.parent.name
            text = load_markdown_file(md_file)
            if not text:
                continue
            raw_chunks = sliding_window_chunks(text)
            for idx, chunk_text in enumerate(raw_chunks):
                chunks.append(Chunk(
                    text=chunk_text,
                    source=str(rel),
                    company=company,
                    product_area=product_area,
                    chunk_index=idx,
                ))

    return chunks


def build_corpus_index() -> Tuple[List[Chunk], Dict]:
    """Load corpus and return chunks + metadata."""
    print(f"[Corpus] Loading documents from {DATA_DIR} ...")
    chunks = discover_corpus()
    print(f"[Corpus] Total chunks: {len(chunks)}")
    metadata = {
        "total_chunks": len(chunks),
        "companies": {},
    }
    for c in chunks:
        metadata["companies"].setdefault(c.company, 0)
        metadata["companies"][c.company] += 1
    for comp, count in metadata["companies"].items():
        print(f"[Corpus]   {comp}: {count} chunks")
    return chunks, metadata


if __name__ == "__main__":
    chunks, meta = build_corpus_index()
    print(meta)

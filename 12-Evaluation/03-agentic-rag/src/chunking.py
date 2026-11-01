# Fixed-size chunking with overlap. Imported by pipeline.py, not run.


def fixed_size_chunk(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunk_size-character pieces; each starts chunk_overlap before the last ended."""
    text = text.strip()
    step = chunk_size - chunk_overlap
    chunks = (text[start:start + chunk_size].strip() for start in range(0, len(text), step))
    return [chunk for chunk in chunks if chunk]

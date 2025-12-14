# chunker.py
"""
Simple chunker: splits text into overlapping chunks by characters.
Produces list of dicts: {'chunk_id', 'start', 'end', 'text'}
"""
def chunk_text(text: str, chunk_size: int = 500, stride: int = 100):
    chunks = []
    start = 0
    chunk_id = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"chunk_id": chunk_id, "start": start, "end": end, "text": chunk})
            chunk_id += 1
        start += (chunk_size - stride)
    return chunks

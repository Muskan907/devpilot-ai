"""Retrieval-quality metrics for the Repository/RAG agent.

Each benchmark task declares the code that is actually relevant to it (ground truth):
files and symbols (functions / classes / methods). Retrieved chunks are scored with
standard IR metrics:

  * precision@k - fraction of the k retrieved chunks that are relevant
  * recall      - fraction of the relevant symbols that appear among the retrieved chunks
  * hit@1       - whether the top-ranked chunk is relevant
  * MRR         - reciprocal rank of the first relevant chunk
"""

from typing import Any, Dict, Iterable, List


def _norm(path: str) -> str:
    return (path or "").replace("\\", "/").lstrip("./")


def _is_relevant(chunk: Dict[str, Any], files: set, symbols: set) -> bool:
    if _norm(chunk.get("file_path", "")) not in files:
        return False
    if not symbols:
        return True
    return bool(_symbol_keys(chunk) & symbols)


def _symbol_keys(chunk: Dict[str, Any]) -> set:
    """A method chunk "Class.method" matches "Class.method" or "method" - not the bare class."""
    name = str(chunk.get("symbol_name", ""))
    return {name, name.split(".")[-1]}


def retrieval_quality(
    chunks: List[Dict[str, Any]],
    relevant_files: Iterable[str],
    relevant_symbols: Iterable[str] = (),
) -> Dict[str, float]:
    files = {_norm(f) for f in relevant_files}
    symbols = set(relevant_symbols)
    k = len(chunks)
    if k == 0:
        return {"k": 0, "precision_at_k": 0.0, "recall": 0.0, "hit_at_1": 0.0, "mrr": 0.0}

    flags = [_is_relevant(c, files, symbols) for c in chunks]
    found = set()
    for c, ok in zip(chunks, flags):
        if ok:
            found |= (_symbol_keys(c) & symbols) if symbols else {_norm(c.get("file_path", ""))}
    denom = len(symbols) if symbols else len(files)
    first = next((i for i, ok in enumerate(flags) if ok), None)
    return {
        "k": k,
        "precision_at_k": round(sum(flags) / k, 3),
        "recall": round(len(found) / denom, 3) if denom else 0.0,
        "hit_at_1": 1.0 if flags[0] else 0.0,
        "mrr": round(1.0 / (first + 1), 3) if first is not None else 0.0,
    }

"""
Chunking module for Docling documents.

Wraps Docling's three native chunkers:
- HierarchicalChunker: Structure-aware, fast, no tokenizer needed
- HybridChunker: Recommended for RAG, tokenizer-aware split/merge
- LineBasedTokenChunker: Preserves line boundaries (tables, code, logs)

Uses chunker.contextualize() to enrich each chunk with structural metadata
(headings, captions, etc.) for better embedding quality.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from docling.chunking import HierarchicalChunker, HybridChunker
from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────

@dataclass
class ChunkItem:
    """A single chunk with text and metadata."""
    index: int
    text: str                     # raw chunk text
    contextualized: str           # text enriched with headings/captions via contextualize()
    num_tokens: int               # approximate token count
    headings: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    chunk_type: str = ""          # e.g. "text", "table", "list_item", "picture"
    page_no: int = 1              # page number in the source document


@dataclass
class ChunkResult:
    """Result of chunking a document with one method."""
    method: str
    method_label: str
    chunks: list[ChunkItem]
    num_chunks: int
    total_tokens: int
    avg_tokens_per_chunk: float
    min_tokens: int
    max_tokens: int
    chunking_time: float


# ──────────────────────────────────────────────
# Chunker Metadata
# ──────────────────────────────────────────────

CHUNKER_INFO = {
    "hierarchical": {
        "label": "HierarchicalChunker",
        "description": "Chunks by document structure (heading, paragraph, list, table). Fast and lightweight.",
        "icon": "🏗️",
        "color": "#34d399",  # emerald
        "supports_max_tokens": False,
        "supports_merge_peers": False,
    },
    "hybrid": {
        "label": "HybridChunker (Recommended)",
        "description": "Combines hierarchical + tokenizer-aware splitting/merging. Best for RAG.",
        "icon": "⚡",
        "color": "#60a5fa",  # blue
        "supports_max_tokens": True,
        "supports_merge_peers": True,
    },
    "line_based": {
        "label": "LineBasedTokenChunker",
        "description": "Preserves line boundaries. Ideal for tables, code, CSV, and logs.",
        "icon": "📏",
        "color": "#fbbf24",  # amber
        "supports_max_tokens": True,
        "supports_merge_peers": False,
    },
}


# ──────────────────────────────────────────────
# Tokenizer Helpers
# ──────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English/mixed text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _get_tokenizer(tokenizer_name: str = "BAAI/bge-m3"):
    """Try to load a HuggingFace tokenizer, return None if unavailable."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        return None


def _count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens using tokenizer if available, else estimate."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return _estimate_tokens(text)


# ──────────────────────────────────────────────
# Extract metadata from chunk
# ──────────────────────────────────────────────

def _extract_chunk_metadata(chunk) -> dict:
    """Extract headings, captions, and type from a Docling BaseChunk."""
    headings = []
    captions = []
    chunk_type = ""
    page_no = 1

    # Extract headings from meta
    if hasattr(chunk, "meta") and chunk.meta:
        meta = chunk.meta
        if hasattr(meta, "headings") and meta.headings:
            headings = list(meta.headings)
        if hasattr(meta, "captions") and meta.captions:
            captions = list(meta.captions)
        if hasattr(meta, "origin"):
            origin = meta.origin
            if hasattr(origin, "content_type"):
                chunk_type = str(origin.content_type)
            elif hasattr(origin, "label"):
                chunk_type = str(origin.label)
        
        # Extract page number from provenance of doc_items
        if hasattr(meta, "doc_items") and meta.doc_items:
            for item in meta.doc_items:
                if hasattr(item, "prov") and item.prov:
                    first_prov = item.prov[0]
                    if hasattr(first_prov, "page_no"):
                        page_no = int(first_prov.page_no)
                        break

    return {
        "headings": headings,
        "captions": captions,
        "chunk_type": chunk_type,
        "page_no": page_no,
    }


# ──────────────────────────────────────────────
# Core Chunking Functions
# ──────────────────────────────────────────────

def run_chunking(
    document,
    method: str = "hybrid",
    max_tokens: int = 512,
    merge_peers: bool = True,
    tokenizer_name: str = "BAAI/bge-m3",
) -> ChunkResult:
    """
    Run chunking on a DoclingDocument using the specified method.

    Uses chunker.contextualize() to produce enriched text for each chunk.

    Args:
        document: DoclingDocument from Docling conversion
        method: "hierarchical", "hybrid", or "line_based"
        max_tokens: Maximum tokens per chunk (Hybrid & LineBased only)
        merge_peers: Merge adjacent small chunks (HybridChunker only)
        tokenizer_name: HuggingFace tokenizer for token counting

    Returns:
        ChunkResult with all chunks and statistics
    """
    start_time = time.time()

    # Create the chunker
    if method == "hierarchical":
        chunker = HierarchicalChunker()
    elif method == "hybrid":
        chunker = HybridChunker(
            tokenizer=tokenizer_name,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
        )
    elif method == "line_based":
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        tokenizer_obj = HuggingFaceTokenizer.from_pretrained(
            model_name=tokenizer_name,
            max_tokens=max_tokens,
        )
        chunker = LineBasedTokenChunker(
            tokenizer=tokenizer_obj,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unknown chunking method: {method}")

    # Run chunking
    raw_chunks = list(chunker.chunk(document))

    # Load tokenizer for accurate counting
    tokenizer = _get_tokenizer(tokenizer_name) if method != "hierarchical" else None

    # Process chunks with contextualize()
    chunk_items = []
    for idx, chunk in enumerate(raw_chunks):
        # Get raw text
        text = chunk.text if hasattr(chunk, "text") else str(chunk)

        # Get contextualized text (enriched with headings/captions)
        try:
            contextualized = chunker.contextualize(chunk)
        except Exception:
            contextualized = text

        # Count tokens
        num_tokens = _count_tokens(contextualized, tokenizer)

        # Extract metadata
        meta = _extract_chunk_metadata(chunk)

        chunk_items.append(ChunkItem(
            index=idx,
            text=text,
            contextualized=contextualized,
            num_tokens=num_tokens,
            headings=meta["headings"],
            captions=meta["captions"],
            chunk_type=meta["chunk_type"],
            page_no=meta["page_no"],
        ))

    elapsed = time.time() - start_time

    # Compute statistics
    token_counts = [c.num_tokens for c in chunk_items]
    total_tokens = sum(token_counts)
    num_chunks = len(chunk_items)

    return ChunkResult(
        method=method,
        method_label=CHUNKER_INFO[method]["label"],
        chunks=chunk_items,
        num_chunks=num_chunks,
        total_tokens=total_tokens,
        avg_tokens_per_chunk=round(total_tokens / num_chunks, 1) if num_chunks > 0 else 0,
        min_tokens=min(token_counts) if token_counts else 0,
        max_tokens=max(token_counts) if token_counts else 0,
        chunking_time=round(elapsed, 3),
    )


def run_all_chunking(
    document,
    max_tokens: int = 512,
    merge_peers: bool = True,
    tokenizer_name: str = "BAAI/bge-m3",
) -> dict[str, ChunkResult]:
    """
    Run all 3 chunking methods on a document for comparison.

    Returns:
        Dict mapping method name to ChunkResult
    """
    results = {}
    for method in ["hierarchical", "hybrid", "line_based"]:
        results[method] = run_chunking(
            document=document,
            method=method,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
            tokenizer_name=tokenizer_name,
        )
    return results

# Docling RAG Pipeline — Architecture & Design

This document describes the technical architecture, models, database design, and search retrieval mechanisms used in the Docling RAG Pipeline.

---

## 1. Pipeline Flow Diagram

The diagram below visualizes the document processing, enrichment, chunking, and ingestion pipeline:

```
                  ┌───────────────────────┐
                  │    Document Input     │
                  │ (PDF, DOCX, PNG, ...) │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   DocumentConverter   │
                  │   (Format Detection)  │
                  └───────────┬───────────┘
                              │
                              ▼
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼ (Standard Mode)                           ▼ (VLM Mode)
┌───────────────────────┐                   ┌───────────────────────┐
│  StandardPdfPipeline  │                   │      VlmPipeline      │
│  (OCR + Layout model) │                   │  (SmolDocling/Vision) │
└───────┬───────────────┘                   └───────────┬───────────┘
        │                                               │
        ├───────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ Markdown + Figures Out  │
└───────┬─────────────────┘
        │
        ▼ (Optional)
┌─────────────────────────┐
│ Qwen3-VL Figure Desc.   │
│   (Enrich Markdown)     │
└───────┬─────────────────┘
        │
        ▼
┌─────────────────────────┐
│    Contextual Chunking  │
│ (Hierarchical/Hybrid)   │
└───────┬─────────────────┘
        │
        ▼
┌─────────────────────────┐
│ BAAI/bge-m3 Embedding   │
│  (1024-dim Normalized)  │
└───────┬─────────────────┘
        │
        ▼
┌─────────────────────────┐
│   PostgreSQL Ingestion  │
│ (pgvector + GIN FTS)    │
└─────────────────────────┘
```

---

## 2. Model Zoo & Specifications

The pipeline integrates several state-of-the-art models for extraction, reasoning, and representation:

| Stage | Model Name | Size / Dim | Key Function | Mode |
|---|---|---|---|---|
| **Layout Analysis** | `Heron` | - | Identifies bounding boxes for headings, text, tables, figures | Standard |
| **OCR Engine** | `RapidOCR` / `EasyOCR` | - | Extracts characters from scanned texts/images | Standard |
| **VLM Converter** | `Granite-Vision` / `SmolDocling` | - | End-to-end vision-based markdown conversion | VLM |
| **Figure Description** | `Qwen/Qwen3-VL-2B-Instruct` | 2.0 B params | Downstream enrichment of figure contents | Standard (Optional) |
| **Text Embedding** | `BAAI/bge-m3` | 1024-dim | Generates normalized dense vectors | Ingestion |

---

## 3. Database Schema

The PostgreSQL database is enabled with the `pgvector` extension and contains two main tables:

### 3.1. `documents`
Stores metadata of processed documents.
```sql
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3.2. `document_chunks`
Stores text segments, contextualized metadata, page numbers, and vector embeddings.
```sql
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    contextualized TEXT NOT NULL,
    page_no INT,
    chunk_type VARCHAR(50),
    headings TEXT[],
    captions TEXT[],
    embedding vector(1024)
);
```

### 3.3. Database Indexes
For fast retrieval at scale, the database maintains two specialized indexes:
1. **Vector Index (HNSW)**: Uses Cosine Distance operator `<=>` for fast dense retrieval.
   ```sql
   CREATE INDEX IF NOT EXISTS document_chunks_hnsw_cosine 
   ON document_chunks USING hnsw (embedding vector_cosine_ops);
   ```
2. **Keyword Index (GIN)**: Fast full-text search indexing on contextualized text using the `'simple'` configuration.
   ```sql
   CREATE INDEX IF NOT EXISTS document_chunks_fts_idx 
   ON document_chunks USING gin (to_tsvector('simple', contextualized));
   ```

---

## 4. Search Retrieval Modes

The system supports three search modalities to compare performance:

### 4.1. Dense Vector Search
Uses cosine similarity to retrieve chunks semantically similar to the user query.
* **Metric**: `1 - (embedding <=> query_vector)`
* **SQL Query**:
  ```sql
  SELECT dc.*, (1 - (dc.embedding <=> %s::vector)) as score 
  FROM document_chunks dc
  ORDER BY dc.embedding <=> %s::vector LIMIT %s;
  ```

### 4.2. Keyword Search (BM25 Equivalent)
Uses PostgreSQL full-text search engine to query tokenized keywords.
* **Metric**: `ts_rank_cd` scoring
* **SQL Query**:
  ```sql
  SELECT dc.*, ts_rank_cd(to_tsvector('simple', dc.contextualized), websearch_to_tsquery('simple', %s)) as score
  FROM document_chunks dc
  WHERE to_tsvector('simple', dc.contextualized) @@ websearch_to_tsquery('simple', %s)
  ORDER BY score DESC LIMIT %s;
  ```

### 4.3. Hybrid Search (RRF)
Combines the strengths of both dense vector search and keyword search. It retrieves top candidates from both searches and ranks them using **Reciprocal Rank Fusion (RRF)**:

$$RRF\_Score(d) = \frac{1}{k + rank_{vector}(d)} + \frac{1}{k + rank_{fts}(d)}$$

Where $k = 60$ (the standard smoothing constant). The item with the highest RRF score is ranked first, neutralizing score calibration issues between dense cosine scores and keyword TF-IDF/BM25 scores.

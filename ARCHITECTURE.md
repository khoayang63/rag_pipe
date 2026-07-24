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
        ├─────────────────────────┬─────────────────────────┐
        ▼ (Text Chunks)           │                         ▼ (Figures/Images)
┌─────────────────────────┐       │                 ┌─────────────────────────┐
│    Contextual Chunking  │       │                 │  BGE-VL-large Embedder  │
│ (Hierarchical/Hybrid)   │       │                 │   (768-dim Normalized)  │
└───────┬─────────────────┘       │                 └───────────┬─────────────┘
        │                         │                             │
        ▼                         │                             ▼
┌─────────────────────────┐       │                 ┌─────────────────────────┐
│ BAAI/bge-m3 Embedding   │       │                 │    document_images      │
│  (1024-dim Normalized)  │       │                 │    PostgreSQL Table     │
└───────┬─────────────────┘       │                 └─────────────────────────┘
        │                         │
        ▼                         │
┌─────────────────────────┐       │
│    document_chunks      │◄──────┘
│    PostgreSQL Table     │
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
| **Text Embedding** | `BAAI/bge-m3` | 1024-dim | Generates normalized dense vectors for text chunks | Ingestion |
| **Visual Embedding** | `BAAI/BGE-VL-large` | 768-dim | Generates joint visual-textual vector embeddings for document figures | Ingestion |

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

### 3.3. `document_images`
Stores cropped figure/diagram images, native captions, VLM visual descriptions, page numbers, and 768-dimensional visual vector embeddings.
```sql
CREATE TABLE IF NOT EXISTS document_images (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) REFERENCES documents(id) ON DELETE CASCADE,
    image_index INT NOT NULL,
    image_path TEXT NOT NULL,
    caption TEXT,
    vlm_description TEXT,
    page_no INT,
    embedding vector(768)
);
```

### 3.4. Database Indexes
For fast retrieval at scale, the database maintains specialized indexes:
1. **Vector Index for Chunks (HNSW)**: Uses Cosine Distance operator `<=>` for fast dense text retrieval.
   ```sql
   CREATE INDEX IF NOT EXISTS document_chunks_hnsw_cosine 
   ON document_chunks USING hnsw (embedding vector_cosine_ops);
   ```
2. **Vector Index for Images (HNSW)**: Uses Cosine Distance operator `<=>` for fast dense visual retrieval on joint embeddings.
   ```sql
   CREATE INDEX IF NOT EXISTS document_images_hnsw_cosine 
   ON document_images USING hnsw (embedding vector_cosine_ops);
   ```
3. **Keyword Index (GIN)**: Fast full-text search indexing on contextualized text using the `'simple'` configuration.
   ```sql
   CREATE INDEX IF NOT EXISTS document_chunks_fts_idx 
   ON document_chunks USING gin (to_tsvector('simple', contextualized));
   ```

---

## 4. Search Retrieval Modes

The system supports four search modalities:

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

### 4.4. Multimodal Image Retrieval
Allows searching for diagrams, charts, and figures directly using joint cross-modal vector mapping.
* **Mechanism**: The user text query is embedded using the text encoder of `BAAI/BGE-VL-large` to produce a 768-dimensional vector, which is then compared against all image embeddings stored in `document_images` using Cosine Distance.
* **Metric**: `1 - (embedding <=> query_vector)`
* **SQL Query**:
  ```sql
  SELECT di.*, (1 - (di.embedding <=> %s::vector)) as score 
  FROM document_images di
  ORDER BY di.embedding <=> %s::vector LIMIT %s;
  ```

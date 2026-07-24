"""
Vector database store module using pgvector in PostgreSQL.
Supports Vector search (cosine distance), BM25 (Full-Text Search), and Hybrid search (RRF).
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Connection parameters with defaults matching docker-compose
DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "ekb")
DB_USER = os.environ.get("POSTGRES_USER", "admin")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "123456")

class VectorStore:
    """Helper class to interact with pgvector and run hybrid search."""
    
    def __init__(self):
        self.conn_params = {
            "host": DB_HOST,
            "port": DB_PORT,
            "database": DB_NAME,
            "user": DB_USER,
            "password": DB_PASS,
            "connect_timeout": 3
        }

    def _get_connection(self):
        """Establish a connection to the database."""
        return psycopg2.connect(**self.conn_params)

    def test_connection(self) -> dict:
        """Test the connection and return status information."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    ver = cur.fetchone()[0]
                    
                    # Attempt to create the extension so it's active immediately
                    try:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        conn.commit()
                    except Exception:
                        conn.rollback()
                    
                    # Check if pgvector is active
                    cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
                    has_vector = cur.fetchone() is not None
                    
                    # If not active, check if it is at least available for creation
                    if not has_vector:
                        cur.execute("SELECT name FROM pg_available_extensions WHERE name = 'vector';")
                        has_vector = cur.fetchone() is not None
                    
                    return {
                        "connected": True,
                        "version": ver,
                        "pgvector_installed": has_vector,
                        "error": None
                    }
        except Exception as e:
            return {
                "connected": False,
                "version": None,
                "pgvector_installed": False,
                "error": str(e)
            }

    def initialize_schema(self):
        """Initialize extensions, tables, and indexes."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                except Exception as e:
                    logger.warning(f"Could not create vector extension (might lack privileges): {e}")
                
                # Create documents metadata table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create chunks table with vector column (dim=1024 for BAAI/bge-m3 dense)
                cur.execute("""
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
                """)
                
                # Create HNSW index for cosine distance similarity search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_hnsw_cosine 
                    ON document_chunks USING hnsw (embedding vector_cosine_ops);
                """)
                
                # Create GIN index for text keyword search (Full-Text Search) using 'simple' configuration
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_fts_idx 
                    ON document_chunks USING gin (to_tsvector('simple', contextualized));
                """)
                
                # Create document_images table with vector column (dim=768 for BAAI/BGE-VL-large dense)
                cur.execute("""
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
                """)
                
                # Create HNSW index for cosine distance similarity search on image embeddings
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_images_hnsw_cosine 
                    ON document_images USING hnsw (embedding vector_cosine_ops);
                """)
            conn.commit()

    def ingest_images(self, doc_id: str, images: List[Dict[str, Any]]) -> int:
        """
        Ingest a document's figures/images with their captions, VLM descriptions, and embeddings.
        Returns the number of images successfully ingested.
        """
        self.initialize_schema()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Clear existing images for this document to avoid duplicates
                cur.execute("DELETE FROM document_images WHERE document_id = %s;", (doc_id,))
                
                # Bulk insert images
                inserted_count = 0
                for img in images:
                    emb = img.get("embedding")
                    if not emb or len(emb) != 768:
                        logger.error(f"Invalid embedding dimensions for image {img.get('index')}: expected 768, got {len(emb) if emb else 0}")
                        continue
                    
                    emb_str = f"[{','.join(map(str, emb))}]"
                    
                    cur.execute("""
                        INSERT INTO document_images 
                        (document_id, image_index, image_path, caption, vlm_description, page_no, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector);
                    """, (
                        doc_id,
                        img["index"],
                        img["image_path"],
                        img.get("caption"),
                        img.get("vlm_description"),
                        img.get("page_no"),
                        emb_str
                    ))
                    inserted_count += 1
                    
            conn.commit()
            return inserted_count

    def image_vector_search(self, query_embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """Perform dense vector search for matching images using Cosine distance."""
        if len(query_embedding) != 768:
            raise ValueError(f"Query embedding size must be 768 for image search, got {len(query_embedding)}")
            
        emb_str = f"[{','.join(map(str, query_embedding))}]"
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        di.id, di.document_id, di.image_index, di.image_path, 
                        di.caption, di.vlm_description, di.page_no,
                        d.name as doc_name,
                        (1 - (di.embedding <=> %s::vector)) as score
                    FROM document_images di
                    JOIN documents d ON di.document_id = d.id
                    ORDER BY di.embedding <=> %s::vector
                    LIMIT %s;
                """, (emb_str, emb_str, limit))
                return [dict(row) for row in cur.fetchall()]

    def ingest_document(self, doc_id: str, doc_name: str, chunks: List[Dict[str, Any]]) -> int:
        """
        Ingest a document metadata and its associated chunks with their embeddings.
        Returns the number of chunks successfully ingested.
        """
        self.initialize_schema()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Insert/Update document metadata
                cur.execute("""
                    INSERT INTO documents (id, name)
                    VALUES (%s, %s)
                    ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name;
                """, (doc_id, doc_name))
                
                # 2. Clear existing chunks for this document to avoid duplicates
                cur.execute("DELETE FROM document_chunks WHERE document_id = %s;", (doc_id,))
                
                # 3. Bulk insert chunks
                inserted_count = 0
                for chunk in chunks:
                    emb = chunk.get("embedding")
                    if not emb or len(emb) != 1024:
                        logger.error(f"Invalid embedding dimensions for chunk {chunk.get('index')}")
                        continue
                    
                    # Convert list of floats to pgvector string format: [0.1,0.2,...]
                    emb_str = f"[{','.join(map(str, emb))}]"
                    
                    cur.execute("""
                        INSERT INTO document_chunks 
                        (document_id, chunk_index, text, contextualized, page_no, chunk_type, headings, captions, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector);
                    """, (
                        doc_id,
                        chunk["index"],
                        chunk["text"],
                        chunk["contextualized"],
                        chunk.get("page_no"),
                        chunk.get("chunk_type", ""),
                        chunk.get("headings", []),
                        chunk.get("captions", []),
                        emb_str
                    ))
                    inserted_count += 1
                    
            conn.commit()
            return inserted_count

    def vector_search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Perform dense vector search using Cosine distance."""
        if len(query_embedding) != 1024:
            raise ValueError(f"Query embedding size must be 1024, got {len(query_embedding)}")
            
        emb_str = f"[{','.join(map(str, query_embedding))}]"
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        dc.id, dc.document_id, dc.chunk_index, dc.text, dc.contextualized, 
                        dc.page_no, dc.chunk_type, dc.headings, dc.captions,
                        d.name as doc_name,
                        (1 - (dc.embedding <=> %s::vector)) as score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    ORDER BY dc.embedding <=> %s::vector
                    LIMIT %s;
                """, (emb_str, emb_str, limit))
                return [dict(row) for row in cur.fetchall()]

    def keyword_search(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform full-text keyword search (BM25 equivalent in PostgreSQL)."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use websearch_to_tsquery for robust handling of search input
                cur.execute("""
                    SELECT 
                        dc.id, dc.document_id, dc.chunk_index, dc.text, dc.contextualized, 
                        dc.page_no, dc.chunk_type, dc.headings, dc.captions,
                        d.name as doc_name,
                        ts_rank_cd(to_tsvector('simple', dc.contextualized), websearch_to_tsquery('simple', %s)) as score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE to_tsvector('simple', dc.contextualized) @@ websearch_to_tsquery('simple', %s)
                    ORDER BY score DESC
                    LIMIT %s;
                """, (query_text, query_text, limit))
                return [dict(row) for row in cur.fetchall()]

    def hybrid_search(self, query_text: str, query_embedding: List[float], limit: int = 5, k: int = 60, use_rerank: bool = False, rerank_top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF).
        Fuses results from Vector Search and BM25 FTS.
        Optionally reranks the top RRF candidates using BGE-Reranker-v2-m3.
        """
        # Fetch more candidates from each search method to perform fusion
        candidate_limit = max(limit * 5, 50)
        if use_rerank:
            candidate_limit = max(candidate_limit, rerank_top_k * 2)
        
        vector_results = self.vector_search(query_embedding, limit=candidate_limit)
        keyword_results = self.keyword_search(query_text, limit=candidate_limit)
        
        rrf_scores = {}
        
        # Helper to index details
        def init_candidate(row):
            return {
                "id": row["id"],
                "document_id": row["document_id"],
                "doc_name": row["doc_name"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "contextualized": row["contextualized"],
                "page_no": row["page_no"],
                "chunk_type": row["chunk_type"],
                "headings": row["headings"],
                "captions": row["captions"],
                "rrf_score": 0.0,
                "vector_rank": None,
                "fts_rank": None,
                "vector_score": None,
                "fts_score": None,
                "rerank_score": None,
            }
            
        # Add vector ranks
        for rank, row in enumerate(vector_results):
            cid = row["id"]
            if cid not in rrf_scores:
                rrf_scores[cid] = init_candidate(row)
            rrf_scores[cid]["vector_rank"] = rank + 1
            rrf_scores[cid]["vector_score"] = float(row["score"])
            rrf_scores[cid]["rrf_score"] += 1.0 / (k + (rank + 1))
                
        # Add keyword FTS ranks
        for rank, row in enumerate(keyword_results):
            cid = row["id"]
            if cid not in rrf_scores:
                rrf_scores[cid] = init_candidate(row)
            rrf_scores[cid]["fts_rank"] = rank + 1
            rrf_scores[cid]["fts_score"] = float(row["score"])
            rrf_scores[cid]["rrf_score"] += 1.0 / (k + (rank + 1))
            
        # Sort by RRF score descending
        sorted_candidates = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        
        if use_rerank and sorted_candidates:
            # Take the top RRF candidates to rerank
            top_rrf = sorted_candidates[:rerank_top_k]
            passages = [c["contextualized"] for c in top_rrf]
            
            # Lazy import BgeReranker to prevent loading during store module import
            from pipeline.processing.reranker import get_reranker
            reranker = get_reranker()
            
            rerank_scores = reranker.compute_scores(query_text, passages)
            for idx, score in enumerate(rerank_scores):
                top_rrf[idx]["rerank_score"] = score
                
            # Sort the reranked subset by rerank_score descending
            sorted_reranked = sorted(top_rrf, key=lambda x: x["rerank_score"], reverse=True)
            return sorted_reranked[:limit]
            
        return sorted_candidates[:limit]

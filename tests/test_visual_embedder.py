"""
Unit tests for the Multimodal Visual Embedder and pgvector database image store.
"""

import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock
import pytest

# Add src to sys.path for direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pipeline.processing.visual_embedder import BgeVisualizedEmbedder
from pipeline.indexer.postgres_store import VectorStore

@patch("pipeline.processing.visual_embedder.AutoModel")
def test_visual_embedder_init(mock_auto_model):
    """Test initializing the visual embedder with mocked model."""
    mock_model = MagicMock()
    # Handle model.to(device) chaining
    mock_model.to.return_value = mock_model
    mock_auto_model.from_pretrained.return_value = mock_model
    import torch
    embedder = BgeVisualizedEmbedder(model_name="BAAI/BGE-VL-large")
    
    mock_auto_model.from_pretrained.assert_called_once_with(
        "BAAI/BGE-VL-large",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    mock_model.set_processor.assert_called_once_with("BAAI/BGE-VL-large")
    mock_model.eval.assert_called_once()

@patch("pipeline.processing.visual_embedder.AutoModel")
def test_embed_text_and_image(mock_auto_model):
    """Test embedding generation for text and images."""
    mock_model = MagicMock()
    # Handle model.to(device) chaining
    mock_model.to.return_value = mock_model
    
    # Mock model.encode to return a numpy array of size 768
    dummy_emb = np.random.rand(768).astype(np.float32)
    mock_model.encode.return_value = dummy_emb
    mock_auto_model.from_pretrained.return_value = mock_model
    
    embedder = BgeVisualizedEmbedder(model_name="BAAI/BGE-VL-large")
    
    # 1. Text embedding
    text_emb = embedder.embed_text("Kiến trúc hệ thống")
    assert len(text_emb) == 768
    mock_model.encode.assert_any_call(text="Kiến trúc hệ thống")
    
    # 2. Image embedding
    img_emb = embedder.embed_image("test_image.png")
    assert len(img_emb) == 768
    mock_model.encode.assert_any_call(images="test_image.png")

def test_postgres_image_ingestion_and_search():
    """Test pgvector ingestion and search operations using mock connections."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    
    # Handle psycopg2 connection context manager yielding mock_conn itself
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    
    # Mock test_connection response queries in VectorStore
    mock_cur.fetchone.return_value = ["PostgreSQL 15"]
    
    db = VectorStore()
    
    # Mock data to ingest
    dummy_emb = [0.1] * 768
    images = [
        {
            "index": 0,
            "image_path": "path/to/fig.png",
            "caption": "Sơ đồ kiến trúc",
            "vlm_description": "Kiến trúc hệ thống bao gồm Frontend và Backend.",
            "page_no": 2,
            "embedding": dummy_emb
        }
    ]
    
    # Patch VectorStore._get_connection directly
    with patch.object(db, "_get_connection", return_value=mock_conn):
        # 1. Test Ingest Images
        count = db.ingest_images(doc_id="test-doc-id", images=images)
        assert count == 1
        
        # Verify insert query was executed
        # Note: DELETE query first, then INSERT query
        mock_cur.execute.assert_any_call("DELETE FROM document_images WHERE document_id = %s;", ("test-doc-id",))
        
        # 2. Test Image Vector Search
        mock_dict_cur = MagicMock()
        mock_dict_cur.fetchall.return_value = [
            {
                "id": 1,
                "document_id": "test-doc-id",
                "image_index": 0,
                "image_path": "path/to/fig.png",
                "caption": "Sơ đồ kiến trúc",
                "vlm_description": "Kiến trúc hệ thống bao gồm Frontend và Backend.",
                "page_no": 2,
                "doc_name": "Tài liệu thiết kế",
                "score": 0.95
            }
        ]
        
        # We reuse the same connection but mock connection cursor manager to yield dict cursor
        mock_conn.cursor.return_value.__enter__.return_value = mock_dict_cur
        
        results = db.image_vector_search(dummy_emb, limit=1)
        
        assert len(results) == 1
        assert results[0]["image_path"] == "path/to/fig.png"
        assert results[0]["score"] == 0.95

"""
Unit tests for the FastAPI Ingest API.
Tests document upload, detection, mocked conversion, and response structures.
"""

import sys
import os
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add src to sys.path for direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from api import app, get_shared_converter


@pytest.fixture
def client():
    return TestClient(app)


@patch("api.get_shared_converter")
@patch("api.detect_input_format")
@patch("api.extract_figures")
def test_ingest_endpoint_pdf(mock_extract_figures, mock_detect_format, mock_get_shared_converter, client):
    """Verify that /ingest correctly handles a PDF and returns markdown and paths."""
    # 1. Setup mocks
    from docling.datamodel.base_models import InputFormat
    mock_detect_format.return_value = InputFormat.PDF
    
    # Mock document conversion result
    mock_result = MagicMock()
    mock_doc = MagicMock()
    mock_result.document = mock_doc
    mock_doc.export_to_markdown.return_value = "# Mocked PDF Content\nThis is a mock PDF conversion."
    
    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result
    mock_get_shared_converter.return_value = mock_converter
    
    # Mock figure extraction
    mock_fig = MagicMock()
    mock_fig.image_path = "scratch/ingest/some-uuid/images/figure_0.png"
    mock_extract_figures.return_value = [mock_fig]
    
    # 2. Call endpoint
    file_content = b"%PDF-1.4 mock pdf content"
    response = client.post(
        "/ingest",
        files={"file": ("test_doc.pdf", file_content, "application/pdf")}
    )
    
    # 3. Assertions
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["filename"] == "test_doc.pdf"
    assert json_data["detected_format"] == "pdf"
    assert json_data["markdown"] == "# Mocked PDF Content\nThis is a mock PDF conversion."
    assert "markdown_file_path" in json_data
    assert "images_directory_path" in json_data
    assert json_data["extracted_images_count"] == 1
    assert json_data["extracted_images"] == ["scratch/ingest/some-uuid/images/figure_0.png"]
    
    # Verify converter.convert was called with a path
    mock_converter.convert.assert_called_once()


@patch("api.detect_input_format")
def test_ingest_unsupported_format(mock_detect_format, client):
    """Verify that /ingest returns 400 for unsupported formats."""
    mock_detect_format.return_value = None
    
    response = client.post(
        "/ingest",
        files={"file": ("test_doc.txt", b"plain text content", "text/plain")}
    )
    
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


@patch("api.get_shared_converter")
@patch("api.detect_input_format")
@patch("api.extract_figures")
def test_ingest_endpoint_xlsx(mock_extract_figures, mock_detect_format, mock_get_shared_converter, client):
    """Verify that /ingest correctly handles an Excel file."""
    from docling.datamodel.base_models import InputFormat
    mock_detect_format.return_value = InputFormat.XLSX
    
    mock_result = MagicMock()
    mock_doc = MagicMock()
    mock_result.document = mock_doc
    mock_doc.export_to_markdown.return_value = "# Mocked Excel Content\n| Col1 | Col2 |\n|---|---|"
    
    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result
    mock_get_shared_converter.return_value = mock_converter
    mock_extract_figures.return_value = []
    
    response = client.post(
        "/ingest",
        files={"file": ("test_doc.xlsx", b"mock xlsx content", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    )
    
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["filename"] == "test_doc.xlsx"
    assert json_data["detected_format"] == "xlsx"
    assert "Mocked Excel Content" in json_data["markdown"]


"""
FastAPI Ingestion Service.
Provides a REST API to ingest documents, convert them to markdown using Docling,
and extract embedded figures to a local directory.
"""

import os
import uuid
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Resolve root directory and insert it into sys.path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.parsers.doc_converter import detect_input_format
from src.pipeline.parsers.figure_extractor import extract_figures

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    DOCLING_LAYOUT_HERON_101,
)

# Initialize FastAPI application
app = FastAPI(
    title="Docling Ingestion Service",
    description="FastAPI REST service to convert documents and extract embedded figures.",
    version="1.0.0",
)

# Base directory for saved artifacts
INGEST_BASE_DIR = ROOT_DIR / "scratch" / "ingest"
INGEST_BASE_DIR.mkdir(parents=True, exist_ok=True)

def get_converter() -> DocumentConverter:
    """
    Configure and instantiate the DocumentConverter.
    PDFs are processed using StandardPdfPipeline with EasyOCR (en, vi) and Heron 101 layout.
    """
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.do_formula_enrichment = True
    options.do_code_enrichment = True
    options.generate_picture_images = True

    # Use EasyOCR with English and Vietnamese
    options.ocr_options = EasyOcrOptions()
    options.ocr_options.lang = ["en", "vi"]

    # Use DOCLING_LAYOUT_HERON_101 for layout detection
    options.layout_options.model_spec = DOCLING_LAYOUT_HERON_101

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=options,
            )
        }
    )
    return converter

# Shared single instance of DocumentConverter
_converter = None

def get_shared_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = get_converter()
    return _converter


@app.post(
    "/ingest",
    summary="Ingest a document, convert to markdown, and extract images",
    response_description="Markdown content and paths to the generated files",
)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document (PDF, DOCX, PPTX, HTML, or Image format).
    The service will:
    1. Detect the file format.
    2. Convert the file to Markdown using Docling Standard Pipeline.
       For PDFs: Uses EasyOCR (EN+VI) and Heron 101 layout model.
    3. Save the Markdown output and extract all embedded figures/images.
    4. Return the Markdown content, path of the saved `.md` file, and path of the images directory.
    """
    # 1. Generate unique request ID and directory
    request_id = str(uuid.uuid4())
    request_dir = INGEST_BASE_DIR / request_id
    request_dir.mkdir(parents=True, exist_ok=True)

    # 2. Save uploaded file locally
    temp_file_path = request_dir / file.filename
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded file: {str(e)}"
        )

    # 3. Detect file format
    input_format = detect_input_format(str(temp_file_path))
    if not input_format:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. Supported extensions: "
                ".pdf, .docx, .pptx, .html, .png, .jpg, .jpeg, .tiff, .bmp, .xlsx, .csv"
            ),
        )

    # 4. Perform conversion using Docling
    try:
        converter = get_shared_converter()
        start_time = time.time()
        result = converter.convert(str(temp_file_path))
        conversion_time = time.time() - start_time

        # Export to markdown
        markdown_content = result.document.export_to_markdown()

        # Save markdown file
        md_filename = Path(file.filename).with_suffix(".md").name
        md_file_path = request_dir / md_filename
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # 5. Extract figures/images
        images_dir = request_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        figures = extract_figures(result.document, str(images_dir))

        # Collect image file paths (as strings)
        image_paths = [fig.image_path for fig in figures]

        return {
            "success": True,
            "filename": file.filename,
            "detected_format": input_format.value,
            "conversion_time_seconds": round(conversion_time, 2),
            "markdown_file_path": str(md_file_path),
            "images_directory_path": str(images_dir),
            "extracted_images_count": len(image_paths),
            "extracted_images": image_paths,
            "markdown": markdown_content,
        }

    except Exception as e:
        import traceback
        print(f"[API Error] {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Error during document conversion: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Start the service on port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

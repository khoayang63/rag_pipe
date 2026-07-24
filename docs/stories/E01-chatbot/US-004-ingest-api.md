# US-004 Ingest API with FastAPI

## Status

in_progress

## Lane

normal

## Product Contract

Expose a REST API endpoint `/ingest` that allows uploading files of various formats (PDF, Word, PPT, HTML, Images), converts them to Markdown using Docling Standard Pipeline (with EasyOCR and Heron 101 for PDFs), extracts the images/figures, and returns the converted markdown content along with the paths to the saved markdown file and extracted images.

## Relevant Product Docs

- [README.md](file:///c:/Users/OS/Desktop/rag_pipeline/README.md)
- [ARCHITECTURE.md](file:///c:/Users/OS/Desktop/rag_pipeline/ARCHITECTURE.md)

## Acceptance Criteria

- Create a FastAPI application running on Uvicorn.
- Provide a `POST /ingest` endpoint that accepts a file upload.
- Automatically detect the file format.
- For PDF files, configure the StandardPdfPipeline to use EasyOCR with English and Vietnamese, and the HERON 101 layout model.
- Save the parsed markdown and extracted figures in a unique subdirectory inside `scratch/ingest/`.
- Return a JSON response containing the markdown string, the saved markdown file path, the extracted images directory path, and a list of extracted image paths.
- Ensure the API is testable via Swagger UI (`/docs`).

## Design Notes

- File: `src/api.py` contains the FastAPI application.
- Uses `docling` and `pydantic` for serialization.
- Uses `extract_figures` from `src/pipeline/parsers/figure_extractor.py`.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-004 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `pytest tests/test_api.py` |
| Integration | Manual request via curl or Swagger UI to `/ingest` |

## Harness Delta

- Added `fastapi` and `uvicorn` backend endpoint to the repository.
- Registered story `US-004` in Harness DB.

## Evidence

- Story registered in Harness DB with ID `US-004`.

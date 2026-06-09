# Docling RAG Pipeline

A standalone web application for document processing using [Docling](https://github.com/DS4SD/docling) with a premium Streamlit UI.

## Features

- **Multi-format input**: PDF, DOCX, PPTX, HTML, PNG, JPG, TIFF
- **Dual pipeline mode**:
  - **Standard**: OCR, table extraction, formula enrichment, code enrichment, figure extraction
  - **VLM**: End-to-end vision-language model conversion (Granite Vision, SmolDocling)
- **Figure extraction**: Automatic extraction and display of figures with metadata (caption, page, bounding box)
- **VLM figure description**: Generate visual descriptions using Qwen3-VL-2B-Instruct
- **Markdown enrichment**: Replace `<!-- image -->` placeholders with figure images + VLM descriptions
- **Pipeline info dashboard**: View models used at each stage, processing times, GPU status

## Setup

### 1. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: For GPU support (required for Qwen3-VL figure description), install the appropriate PyTorch version for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Set environment variable

```bash
# Windows (Command Prompt)
set HF_TOKEN=hf_your_token_here

# Windows (PowerShell)
$env:HF_TOKEN = "hf_your_token_here"

# Linux / macOS
export HF_TOKEN=hf_your_token_here
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

### 4. Run the application

```bash
streamlit run src/app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

### 1. Web Application (Streamlit)

Run the following command to start the web interface:
```bash
streamlit run src/app.py
```
1. **Configure pipeline** — Use the sidebar to choose pipeline mode and options
2. **Upload document** — Drag and drop or click to browse
3. **Convert** — Click "Convert Document" to process
4. **View results** — Browse Markdown Output, Extracted Figures, and Pipeline Info tabs
5. **Describe figures** (optional) — Click "Describe Figures" to generate VLM descriptions (requires GPU)

### 2. Command Line Interface (CLI)

Use the CLI script to process documents directly from your terminal. This is highly recommended if you are downloading large models (like the built-in description model) to see real-time progress bars (tqdm) and avoid Web UI freezes.

```bash
# Basic usage (Standard pipeline: OCR, tables, formulas, code, figure extraction)
python src/cli.py path/to/document.pdf

# Enable Docling's built-in picture description (downloads weights with terminal progress bar)
python src/cli.py path/to/document.pdf --built-in-desc

# Run standard pipeline and enrich markdown downstream using Qwen3-VL-2B-Instruct (GPU required)
python src/cli.py path/to/document.pdf --enrich

# Run standard pipeline and enrich markdown downstream using a super light model like SmolVLM
python src/cli.py path/to/document.pdf --enrich --desc-model HuggingFaceTB/SmolVLM-256M-Instruct

# Run end-to-end VLM pipeline using Granite Docling VLM model (258M parameter)
python src/cli.py path/to/document.pdf --mode vlm --vlm-preset GRANITEDOCLING_TRANSFORMERS

# Specify a custom output directory (Default is cli_output/)
python src/cli.py path/to/document.pdf -o my_output_dir
```

#### CLI Command Options:
* `input`: Path to input document (required).
* `--mode {standard,vlm}`: Pipeline mode (default: `standard`).
* `--no-ocr`: Disable OCR.
* `--no-table`: Disable table structure extraction.
* `--no-formula`: Disable formula enrichment.
* `--no-code`: Disable code block extraction.
* `--built-in-desc`: Enable Docling's built-in picture description model.
* `--no-pic-extract`: Disable exporting cropped images of figures.
* `--image-scale`: OCR image zoom scale (default: `1.0`).
* `--vlm-preset`: VLM model preset for VLM mode (`GRANITE_VISION_TRANSFORMERS`, `GRANITEDOCLING_TRANSFORMERS`, `SMOLDOCLING_TRANSFORMERS`).
* `--enrich` (or `--use-qwen`): Generate figure descriptions downstream and enrich markdown.
* `--desc-model` (or `--qwen-model`): Vision model ID for downstream description (default: `Qwen/Qwen3-VL-2B-Instruct`).
* `-o`, `--output-dir`: Directory to save outputs (default: `cli_output`).
* `-h`, `--help`: Show help message and options.

## Project Structure

```
src/
├── app.py                    # Streamlit entry point
├── config.py                 # Environment config (HF_TOKEN, paths)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── pipeline/
│   ├── __init__.py
│   ├── converter.py          # Docling DocumentConverter setup
│   ├── figure_extractor.py   # Figure extraction + image saving
│   ├── vlm_describer.py      # Qwen3-VL figure description
│   └── enrichment.py         # Markdown enrichment
└── ui/
    ├── __init__.py
    ├── styles.py             # Custom CSS (dark theme)
    ├── sidebar.py            # Pipeline config sidebar
    ├── upload.py             # File upload component
    ├── markdown_viewer.py    # Markdown viewer (rendered + raw)
    ├── figure_gallery.py     # Figure grid gallery
    └── pipeline_info.py      # Pipeline architecture info
```

## Pipeline Architecture

```
Document Input
  ↓
DocumentConverter (format detection)
  ↓
StandardPdfPipeline / VlmPipeline
  ↓
OCR → Table → Formula → Code → Figure Extraction
  ↓
Markdown Export
  ↓
Qwen3-VL Figure Description (optional, GPU)
  ↓
Markdown Enrichment
  ↓
Ready for RAG (Chunking → Embedding)
```

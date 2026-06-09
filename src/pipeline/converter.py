"""
Document Converter module.

Sets up Docling DocumentConverter with either StandardPdfPipeline or VlmPipeline.
Supports multiple input formats: PDF, DOCX, PPTX, HTML, IMAGE.
"""

import gc
import time
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

import docling.utils.profiling as profiling
profiling.settings.debug.profile_pipeline_timings = True


@dataclass
class PipelineConfig:
    """Configuration for the document conversion pipeline."""

    # Pipeline mode
    pipeline_mode: str = "standard"  # "standard" or "vlm"

    # Standard pipeline options
    do_ocr: bool = True
    do_table_structure: bool = True
    do_formula_enrichment: bool = True
    do_code_enrichment: bool = True
    do_picture_description: bool = False
    do_picture_classification: bool = False
    generate_page_images: bool = False
    generate_picture_images: bool = True
    images_scale: float = 1.0

    # OCR options
    ocr_languages: list = field(default_factory=lambda: ["en"])
    ocr_engine: str = "easyocr"

    # Layout options
    layout_model: str = "layout_v2"

    # VLM options (when pipeline_mode == "vlm")
    vlm_preset: str = "GRANITE_VISION_TRANSFORMERS"
    vlm_scale: float = 1.0
    vlm_max_size: int = 1024
    vlm_max_new_tokens: int = 512

    # Downstream vision model for figure description
    downstream_model: str = "Qwen/Qwen3-VL-2B-Instruct"


@dataclass
class ConversionResult:
    """Result of a document conversion."""

    document: object  # docling Document
    markdown: str
    conversion_time: float
    pipeline_mode: str
    pipeline_config: dict
    input_format: str
    timings: Optional[dict] = None


def create_standard_converter(config: PipelineConfig) -> DocumentConverter:
    """
    Create a DocumentConverter using StandardPdfPipeline.

    Per Docling Skill: StandardPdfPipeline is the default for
    OCR, Layout Analysis, Table/Formula/Figure Extraction, and RAG.
    """
    options = PdfPipelineOptions()

    options.do_ocr = config.do_ocr
    options.do_table_structure = config.do_table_structure
    options.do_formula_enrichment = config.do_formula_enrichment
    options.do_code_enrichment = config.do_code_enrichment
    options.do_picture_description = config.do_picture_description
    options.do_picture_classification = config.do_picture_classification
    options.generate_page_images = config.generate_page_images
    options.generate_picture_images = config.generate_picture_images
    options.images_scale = config.images_scale

    # Set OCR engine options
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        RapidOcrOptions,
        TesseractOcrOptions,
        TesseractCliOcrOptions,
        OcrMacOptions,
        KserveV2OcrOptions,
        OcrAutoOptions,
    )

    ocr_map = {
        "easyocr": EasyOcrOptions,
        "rapidocr": RapidOcrOptions,
        "tesseract": TesseractOcrOptions,
        "tesseract_cli": TesseractCliOcrOptions,
        "macocr": OcrMacOptions,
        "kserve": KserveV2OcrOptions,
        "auto": OcrAutoOptions,
    }

    ocr_cls = ocr_map.get(config.ocr_engine.lower(), EasyOcrOptions)
    options.ocr_options = ocr_cls()

    if hasattr(options.ocr_options, "lang"):
        options.ocr_options.lang = config.ocr_languages

    # Set Layout model options
    from docling.datamodel.pipeline_options import (
        DOCLING_LAYOUT_V2,
        DOCLING_LAYOUT_HERON,
        DOCLING_LAYOUT_HERON_101,
        DOCLING_LAYOUT_EGRET_MEDIUM,
        DOCLING_LAYOUT_EGRET_LARGE,
        DOCLING_LAYOUT_EGRET_XLARGE,
    )

    layout_map = {
        "layout_v2": DOCLING_LAYOUT_V2,
        "heron": DOCLING_LAYOUT_HERON,
        "heron_101": DOCLING_LAYOUT_HERON_101,
        "egret_medium": DOCLING_LAYOUT_EGRET_MEDIUM,
        "egret_large": DOCLING_LAYOUT_EGRET_LARGE,
        "egret_xlarge": DOCLING_LAYOUT_EGRET_XLARGE,
    }

    layout_spec = layout_map.get(config.layout_model.lower(), DOCLING_LAYOUT_V2)
    options.layout_options.model_spec = layout_spec

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=options,
            )
        }
    )

    return converter


def create_vlm_converter(config: PipelineConfig) -> DocumentConverter:
    """
    Create a DocumentConverter using VlmPipeline.

    Per Docling Skill: VlmPipeline is used only when end-to-end VLM
    conversion is requested or the document is image-heavy.
    """
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel import vlm_model_specs
    from docling.pipeline.vlm_pipeline import VlmPipeline

    # Get the VLM preset
    preset_map = {
        "GRANITE_VISION_TRANSFORMERS": vlm_model_specs.GRANITE_VISION_TRANSFORMERS,
        "GRANITEDOCLING_TRANSFORMERS": vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
        "SMOLDOCLING_TRANSFORMERS": vlm_model_specs.SMOLDOCLING_TRANSFORMERS,
    }

    preset = preset_map.get(
        config.vlm_preset,
        vlm_model_specs.GRANITE_VISION_TRANSFORMERS,
    )

    opt = deepcopy(preset)
    opt.scale = config.vlm_scale
    opt.max_size = config.vlm_max_size
    opt.max_new_tokens = config.vlm_max_new_tokens

    pipeline_options = VlmPipelineOptions(vlm_options=opt)
    pipeline_options.images_scale = config.images_scale

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    return converter


def detect_input_format(file_path: str) -> Optional[InputFormat]:
    """Detect the InputFormat from file extension."""
    ext = Path(file_path).suffix.lower()
    format_map = {
        ".pdf": InputFormat.PDF,
        ".docx": InputFormat.DOCX,
        ".pptx": InputFormat.PPTX,
        ".html": InputFormat.HTML,
        ".htm": InputFormat.HTML,
        ".png": InputFormat.IMAGE,
        ".jpg": InputFormat.IMAGE,
        ".jpeg": InputFormat.IMAGE,
        ".tiff": InputFormat.IMAGE,
        ".tif": InputFormat.IMAGE,
        ".bmp": InputFormat.IMAGE,
    }
    return format_map.get(ext)


def convert_document(
    file_path: str,
    config: PipelineConfig,
) -> ConversionResult:
    """
    Convert a document using the configured pipeline.

    Returns a ConversionResult with the document, markdown, and metadata.
    """
    # Clear GPU cache before conversion
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        gc.collect()

    # Create the appropriate converter
    if config.pipeline_mode == "vlm":
        converter = create_vlm_converter(config)
    else:
        converter = create_standard_converter(config)

    # Run conversion
    start_time = time.time()
    result = converter.convert(file_path)
    conversion_time = time.time() - start_time

    # Export to markdown
    md = result.document.export_to_markdown()

    # Build config dict for display
    if config.pipeline_mode == "vlm":
        config_dict = {
            "pipeline": "VlmPipeline",
            "vlm_preset": config.vlm_preset,
            "vlm_scale": config.vlm_scale,
            "vlm_max_size": config.vlm_max_size,
            "vlm_max_new_tokens": config.vlm_max_new_tokens,
            "images_scale": config.images_scale,
        }
    else:
        config_dict = {
            "pipeline": "StandardPdfPipeline",
            "do_ocr": config.do_ocr,
            "ocr_engine": config.ocr_engine,
            "layout_model": config.layout_model,
            "do_table_structure": config.do_table_structure,
            "do_formula_enrichment": config.do_formula_enrichment,
            "do_code_enrichment": config.do_code_enrichment,
            "do_picture_description": config.do_picture_description,
            "do_picture_classification": config.do_picture_classification,
            "generate_page_images": config.generate_page_images,
            "generate_picture_images": config.generate_picture_images,
            "images_scale": config.images_scale,
        }

    input_format = detect_input_format(file_path)
    format_name = input_format.value if input_format else "unknown"

    return ConversionResult(
        document=result.document,
        markdown=md,
        conversion_time=conversion_time,
        pipeline_mode=config.pipeline_mode,
        pipeline_config=config_dict,
        input_format=format_name,
        timings=result.timings,
    )

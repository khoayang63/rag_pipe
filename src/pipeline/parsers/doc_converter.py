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
    ocr_engine: str = "rapidocr"

    # Layout options
    layout_model: str = "heron"

    # VLM options (when pipeline_mode == "vlm")
    vlm_preset: str = "GRANITE_VISION_TRANSFORMERS"
    vlm_scale: float = 1.0
    vlm_max_size: int = 1024
    vlm_max_new_tokens: int = 512

    # Downstream vision model for figure description
    downstream_model: str = "Qwen/Qwen3-VL-2B-Instruct"

    # PDF splitting options
    enable_pdf_splitting: bool = True
    pdf_splitting_part_size: int = 15


@dataclass
class ConversionResult:
    """Result of a document conversion."""

    document: object  # docling Document
    markdown: str
    conversion_time: float
    pipeline_mode: str
    pipeline_config: dict
    input_format: str
    source_filename: str = ""
    timings: Optional[dict] = None


EASYOCR_GROUPS = {
    "chinese_sim": {"ch_sim"},
    "chinese_tra": {"ch_tra"},
    "japanese": {"ja"},
    "korean": {"ko"},
    "thai": {"th"},
    "tamil": {"ta"},
    "telugu": {"te"},
    "kannada": {"kn"},
    "bengali": {"bn", "as"},
    "arabic": {"ar", "fa", "ur", "ug"},
    "devanagari": {"hi", "mr", "ne"},
    "cyrillic": {"ru", "rs_cyrillic", "be", "bg", "uk", "mn"},
    "latin": {
        "af", "az", "bs", "cs", "cy", "da", "de", "es", "et", "fr", "ga", "hr", "hu",
        "id", "is", "it", "ku", "la", "lt", "lv", "mi", "ms", "mt", "nl", "no", "oc",
        "pi", "pl", "pt", "ro", "rs_latin", "sk", "sl", "sq", "sv", "sw", "tl", "tr",
        "uz", "vi"
    }
}


def _filter_easyocr_compatibility(langs: list) -> list:
    """
    Filter the language list to ensure it only contains mutually compatible
    languages for EasyOCR, preventing compatibility ValueErrors.
    """
    # Find the first language in langs that is NOT "en"
    first_non_en = None
    for lang in langs:
        if lang != "en":
            first_non_en = lang
            break

    if not first_non_en:
        return langs

    # Find the active group for first_non_en
    active_group_set = set()
    for name, g_set in EASYOCR_GROUPS.items():
        if first_non_en in g_set:
            active_group_set = g_set
            break

    if not active_group_set:
        # Fallback: keep only "en" and the unrecognized language
        return [lang for lang in langs if lang == "en" or lang == first_non_en]

    # Keep only "en" and languages belonging to the active group
    filtered = []
    for lang in langs:
        if lang == "en" or lang in active_group_set:
            filtered.append(lang)

    return filtered


def _map_ocr_languages(ocr_languages: list, engine: str) -> list:
    """
    Map UI/standard language codes to engine-specific codes.
    Supported UI inputs: ["en", "vi", "zh", "ja", "ko", "fr", "de", "es", "pt", "it", "ru", "ar"]
    """
    if not ocr_languages:
        return ocr_languages

    engine_lower = engine.lower()

    if engine_lower == "easyocr":
        mapped = []
        for lang in ocr_languages:
            if lang == "zh":
                # Default "zh" to "ch_sim" for simplified Chinese
                mapped.append("ch_sim")
            else:
                mapped.append(lang)
        # Deduplicate preserving order
        seen = set()
        deduped = [x for x in mapped if not (x in seen or seen.add(x))]
        return _filter_easyocr_compatibility(deduped)

    elif engine_lower in ("tesseract", "tesseract_cli"):
        tess_map = {
            "en": "eng",
            "vi": "vie",
            "zh": "chi_sim",
            "ja": "jpn",
            "ko": "kor",
            "fr": "fra",
            "de": "deu",
            "es": "spa",
            "pt": "por",
            "it": "ita",
            "ru": "rus",
            "ar": "ara"
        }
        mapped = []
        for lang in ocr_languages:
            if lang in tess_map:
                mapped.append(tess_map[lang])
            else:
                mapped.append(lang)
        seen = set()
        return [x for x in mapped if not (x in seen or seen.add(x))]

    elif engine_lower == "rapidocr":
        mapped = []
        for lang in ocr_languages:
            if lang == "en":
                mapped.append("english")
            elif lang == "zh":
                mapped.append("chinese")
            else:
                mapped.append(lang)
        return mapped

    elif engine_lower == "macocr":
        mac_map = {
            "en": "en-US",
            "vi": "vi-VN",
            "zh": "zh-CN",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "fr": "fr-FR",
            "de": "de-DE",
            "es": "es-ES",
            "pt": "pt-PT",
            "it": "it-IT",
            "ru": "ru-RU",
            "ar": "ar-SA"
        }
        mapped = []
        for lang in ocr_languages:
            if lang in mac_map:
                mapped.append(mac_map[lang])
            else:
                mapped.append(lang)
        return mapped

    return ocr_languages


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

    import sys
    selected_ocr = config.ocr_engine.lower()
    if selected_ocr == "macocr" and sys.platform != "darwin":
        selected_ocr = "rapidocr"

    ocr_map = {
        "easyocr": EasyOcrOptions,
        "rapidocr": RapidOcrOptions,
        "tesseract": TesseractOcrOptions,
        "tesseract_cli": TesseractCliOcrOptions,
        "macocr": OcrMacOptions,
        "kserve": KserveV2OcrOptions,
        "auto": OcrAutoOptions,
    }

    ocr_cls = ocr_map.get(selected_ocr, EasyOcrOptions)
    options.ocr_options = ocr_cls()

    if hasattr(options.ocr_options, "lang"):
        options.ocr_options.lang = _map_ocr_languages(config.ocr_languages, selected_ocr)

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
        ".xlsx": InputFormat.XLSX,
        ".csv": InputFormat.CSV,
    }
    return format_map.get(ext)


def _detect_actual_ocr(converter) -> str:
    """Detect the actual OCR engine class initialized in the converter."""
    try:
        ocr_model = None
        for k, pipeline in converter.initialized_pipelines.items():
            if hasattr(pipeline, "ocr_model") and pipeline.ocr_model is not None:
                ocr_model = pipeline.ocr_model
                break
        
        if ocr_model:
            if type(ocr_model).__name__ == "OcrAutoModel" and hasattr(ocr_model, "_engine") and ocr_model._engine is not None:
                ocr_model = ocr_model._engine
            
            ocr_model_class = type(ocr_model).__name__
            if "EasyOcr" in ocr_model_class:
                return "easyocr"
            elif "RapidOcr" in ocr_model_class:
                return "rapidocr"
            elif "Tesseract" in ocr_model_class:
                return "tesseract"
            elif "MacOcr" in ocr_model_class:
                return "macocr"
            return ocr_model_class.lower()
    except Exception:
        pass
    return "unknown"



def convert_document(
    file_path: str,
    config: PipelineConfig,
    progress_callback=None,
) -> ConversionResult:
    """
    Convert a document using the configured pipeline.
    If the document is a PDF with more than part_size pages, it is processed
    sequentially in parts of 10-20 pages each to prevent Out of Memory (std::bad_alloc).
    """
    # Clear GPU cache before conversion
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        gc.collect()

    input_format = detect_input_format(file_path)
    format_name = input_format.value if input_format else "unknown"

    # Detect if PDF splitting is needed
    is_large_pdf = False
    total_pages = 0
    part_size = getattr(config, "pdf_splitting_part_size", 15)
    enable_pdf_splitting = getattr(config, "enable_pdf_splitting", True)

    if input_format == InputFormat.PDF and enable_pdf_splitting:
        try:
            import pypdfium2 as pdfium
            with pdfium.PdfDocument(file_path) as pdf:
                total_pages = len(pdf)
            if total_pages > part_size:
                is_large_pdf = True
        except Exception as e:
            print(f"[doc_converter] Error checking PDF page count: {e}")

    # Create the appropriate converter
    if config.pipeline_mode == "vlm":
        converter = create_vlm_converter(config)
    else:
        converter = create_standard_converter(config)

    start_time = time.time()

    if is_large_pdf:
        # Sequential split processing
        from docling_core.types.doc import DoclingDocument
        docs_to_concat = []

        split_msg = f"PDF has {total_pages} pages. Splitting into parts of {part_size} pages..."
        print(f"[doc_converter] {split_msg}")
        if progress_callback:
            progress_callback(split_msg)

        for start in range(1, total_pages + 1, part_size):
            end = min(start + part_size - 1, total_pages)
            
            # Clear cache before each part
            try:
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except ImportError:
                gc.collect()
                
            range_msg = f"Processing page range {start}-{end} of {total_pages}..."
            print(f"[doc_converter] {range_msg}")
            if progress_callback:
                progress_callback(range_msg)

            part_result = converter.convert(file_path, page_range=(start, end))
            docs_to_concat.append(part_result.document)

        concat_msg = "Concatenating page ranges..."
        print(f"[doc_converter] {concat_msg}")
        if progress_callback:
            progress_callback(concat_msg)

        document = DoclingDocument.concatenate(docs=docs_to_concat)
        conversion_time = time.time() - start_time
        md = document.export_to_markdown()
        timings = None
    else:
        # Normal single conversion
        result = converter.convert(file_path)
        conversion_time = time.time() - start_time
        document = result.document
        md = document.export_to_markdown()
        timings = result.timings

    # Build config dict for display
    config_dict = _build_config_dict(config)
    config_dict["actual_ocr"] = _detect_actual_ocr(converter)
    if is_large_pdf:
        config_dict["split_processing"] = f"Yes ({part_size}-page chunks)"

    return ConversionResult(
        document=document,
        markdown=md,
        conversion_time=conversion_time,
        pipeline_mode=config.pipeline_mode,
        pipeline_config=config_dict,
        input_format=format_name,
        source_filename=Path(file_path).name,
        timings=timings,
    )


def _build_config_dict(config: PipelineConfig) -> dict:
    """Build a display-friendly config dict from PipelineConfig."""
    if config.pipeline_mode == "vlm":
        return {
            "pipeline": "VlmPipeline",
            "vlm_preset": config.vlm_preset,
            "vlm_scale": config.vlm_scale,
            "vlm_max_size": config.vlm_max_size,
            "vlm_max_new_tokens": config.vlm_max_new_tokens,
            "images_scale": config.images_scale,
        }
    return {
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
        "downstream_model": config.downstream_model,
    }


def convert_documents(
    file_paths: list[str],
    config: PipelineConfig,
    progress_callback=None,
) -> list[ConversionResult]:
    """
    Batch-convert multiple documents.
    Calls convert_document sequentially for each file, which handles
    PDF page-splitting and clears VRAM/RAM between documents.
    """
    results = []
    total_files = len(file_paths)
    
    for i, file_path in enumerate(file_paths):
        filename = Path(file_path).name
        if progress_callback:
            progress_callback(i + 1, total_files, filename)
        
        # Clear cache between documents
        try:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except ImportError:
            gc.collect()
            
        # Define local callback to report internal page conversion status
        local_cb = None
        if progress_callback:
            local_cb = lambda msg: progress_callback(i + 1, total_files, f"{filename} ({msg})")
            
        results.append(convert_document(file_path, config, progress_callback=local_cb))
        
    return results


#!/usr/bin/env python
"""
Docling RAG Pipeline — CLI Inference Script

Enables command-line document conversion, figure extraction, and VLM enrichment.
Shows real-time progress bars (tqdm) for HuggingFace model downloads, preventing UI freezes.

Usage:
  python src/cli.py <path_to_document> [options]
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Configure utf-8 encoding on Windows to prevent UnicodeEncodeError in terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Add src folder to sys.path for direct imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from config import get_hf_token, login_huggingface, check_gpu_available
from pipeline.converter import PipelineConfig, convert_document, detect_input_format
from pipeline.figure_extractor import extract_figures
from pipeline.enrichment import enrich_markdown, count_image_placeholders, save_markdown
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Docling RAG Pipeline CLI — Run document conversion via Terminal."
    )
    
    # Required arguments
    parser.add_argument(
        "input",
        type=str,
        help="Path to input document (PDF, DOCX, PPTX, HTML, PNG, JPG, ...)"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["standard", "vlm"],
        default="standard",
        help="Pipeline Mode: 'standard' (OCR + Layout, recommended for RAG) or 'vlm' (Vision Language Model)."
    )
    
    # Pipeline options (Standard mode)
    parser.add_argument(
        "--no-ocr",
        action="store_false",
        dest="do_ocr",
        help="Disable Optical Character Recognition (OCR)."
    )
    parser.add_argument(
        "--no-table",
        action="store_false",
        dest="do_table",
        help="Disable table structure extraction."
    )
    parser.add_argument(
        "--no-formula",
        action="store_false",
        dest="do_formula",
        help="Disable formula extraction/enrichment."
    )
    parser.add_argument(
        "--no-code",
        action="store_false",
        dest="do_code",
        help="Disable code block extraction."
    )
    parser.add_argument(
        "--built-in-desc",
        action="store_true",
        dest="do_picture_description",
        help="Enable Docling's built-in picture description model (Will download heavy model weights in terminal)."
    )
    parser.add_argument(
        "--no-pic-extract",
        action="store_false",
        dest="generate_picture_images",
        help="Disable exporting cropped images of figures."
    )
    parser.add_argument(
        "--pic-class",
        action="store_true",
        dest="do_picture_classification",
        help="Enable picture classification model (Will classify figures into chart, diagram, etc.)."
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help="OCR Image zoom scale (Default: 1.0. Set to 2.0 for scanned/tiny text)."
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["easyocr", "rapidocr", "tesseract", "tesseract_cli", "macocr", "kserve", "auto"],
        default="easyocr",
        help="OCR engine to use for standard pipeline (Default: easyocr)."
    )
    parser.add_argument(
        "--layout-model",
        choices=["layout_v2", "heron", "heron_101", "egret_medium", "egret_large", "egret_xlarge"],
        default="layout_v2",
        help="Layout analysis model to use for standard pipeline (Default: layout_v2)."
    )
    
    # VLM options
    parser.add_argument(
        "--vlm-preset",
        choices=["GRANITE_VISION_TRANSFORMERS", "GRANITEDOCLING_TRANSFORMERS", "SMOLDOCLING_TRANSFORMERS"],
        default="GRANITEDOCLING_TRANSFORMERS",
        help="VLM model preset when using --mode vlm (Default: GRANITEDOCLING_TRANSFORMERS)."
    )
    
    # Downstream VLM Enrichment options
    parser.add_argument(
        "--enrich", "--use-qwen",
        action="store_true",
        dest="enrich",
        help="Generate figure descriptions downstream using a vision model and enrich markdown."
    )
    parser.add_argument(
        "--desc-model", "--qwen-model",
        type=str,
        dest="qwen_model",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model ID for downstream description (e.g., 'Qwen/Qwen3-VL-2B-Instruct', 'HuggingFaceTB/SmolVLM-256M-Instruct'. Default: Qwen/Qwen3-VL-2B-Instruct)."
    )
    
    # Output path
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="cli_output",
        help="Directory to save outputs (Default: 'cli_output')."
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Validate Input File
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n[ERROR] File does not exist: {input_path}")
        sys.exit(1)
        
    print("=" * 65)
    print("         DOCLING RAG PIPELINE — TERMINAL INFERENCE")
    print("=" * 65)
    print(f"Input file : {input_path.resolve()}")
    print(f"Pipeline   : {args.mode.upper()}")
    print(f"Output dir : {Path(args.output_dir).resolve()}")
    print("-" * 65)
    
    # 2. HuggingFace Login
    try:
        token = get_hf_token()
        print(f"HF Token found: {token[:6]}...{token[-4:] if len(token) > 10 else ''}")
        print("Logging in to HuggingFace Hub...")
        if login_huggingface(token):
            print("[SUCCESS] HuggingFace Hub login OK!")
        else:
            print("[WARNING] HuggingFace login might have failed. Continuing anyway...")
    except EnvironmentError as e:
        print(f"\n[WARNING] HuggingFace Token not set in env: {e}")
        print("Continuing... (downloads of gated models might fail without token)")
    
    # 3. Create Pipeline Config
    config = PipelineConfig(
        pipeline_mode=args.mode,
        do_ocr=args.do_ocr,
        ocr_engine=args.ocr_engine,
        layout_model=args.layout_model,
        do_table_structure=args.do_table,
        do_formula_enrichment=args.do_formula,
        do_code_enrichment=args.do_code,
        do_picture_description=args.do_picture_description,
        do_picture_classification=args.do_picture_classification,
        generate_page_images=False, # Avoid RAM OOM/bad_alloc
        generate_picture_images=args.generate_picture_images,
        images_scale=args.image_scale,
        vlm_preset=args.vlm_preset
    )
    
    # 4. Check GPU status
    gpu_info = check_gpu_available()
    print(f"System GPU : {'AVAILABLE' if gpu_info['available'] else 'NOT AVAILABLE'} ({gpu_info['device']})")
    
    # 5. Run Document Conversion
    print("\n[STEP 1/3] Converting document using Docling...")
    if args.do_picture_description:
        print("[NOTE] Built-in Picture Description is ENABLED.")
        print("[NOTE] If running for the first time, terminal will display downloading progress bars.")
    
    start_time = time.time()
    try:
        result = convert_document(str(input_path), config)
        elapsed = time.time() - start_time
        print(f"[SUCCESS] Conversion completed in {elapsed:.2f} seconds.")
    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # Prepare output directories
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 6. Extract Figure Images (if enabled)
    figures = []
    if args.generate_picture_images:
        print("\n[STEP 2/3] Extracting figure images...")
        try:
            figures = extract_figures(result.document, str(figures_dir))
            print(f"[SUCCESS] Extracted {len(figures)} figure images into: {figures_dir}")
            
            # Print figure classifications if enabled
            if args.do_picture_classification and figures:
                print("\n--- Figure Classification Results ---")
                for fig in figures:
                    if fig.classification:
                        print(f"Figure {fig.index + 1}:")
                        print(f"  Type: {fig.classification}")
                        print(f"  Confidence: {fig.confidence:.4f}")
                    else:
                        print(f"Figure {fig.index + 1}: No classification data available.")
                print("-" * 37)
        except Exception as e:
            print(f"[ERROR] Figure extraction failed: {e}")
            
    # 7. Post-processing: Qwen VLM Enrichment (if requested)
    enriched_md = None
    if args.enrich and figures:
        if not gpu_info["available"]:
            print(f"\n[WARNING] CUDA is not available. Skipping downstream VLM enrichment.")
        else:
            print(f"\n[STEP 3/3] Running downstream figure description using {args.qwen_model}...")
            from pipeline.vlm_describer import describe_figures, load_model
            
            print(f"Loading {args.qwen_model} model weights...")
            model, processor = load_model(model_id=args.qwen_model)
            
            if model is not None:
                print(f"Describing {len(figures)} figures...")
                desc_result = describe_figures(figures, model=model, processor=processor, model_id=args.qwen_model)
                
                print("Enriching markdown output with descriptions...")
                figure_paths = [fig.image_path for fig in figures]
                enriched_md = enrich_markdown(
                    result.markdown,
                    desc_result.descriptions,
                    figure_paths,
                    figures_base_dir="figures"
                )
                print(f"[SUCCESS] Downstream VLM inference completed in {desc_result.inference_time:.2f}s.")
            else:
                print(f"[ERROR] Failed to load {args.qwen_model} model.")
    
    # 8. Save output markdown
    final_md = enriched_md if enriched_md is not None else result.markdown
    output_md_path = out_dir / f"{input_path.stem}.md"
    save_markdown(final_md, str(output_md_path))

    # Print Docling conversion timings breakdown
    if result.timings:
        print("\n" + "=" * 65)
        print("                  DOCLING PIPELINE STAGE TIMINGS")
        print("=" * 65)
        rows = []
        for name, item in result.timings.items():
            total = sum(item.times)
            avg = total / item.count
            rows.append([
                name,
                item.scope.value,
                item.count,
                f"{total:.2f}s",
                f"{avg:.2f}s"
            ])
        rows.sort(key=lambda x: float(x[3][:-1]), reverse=True)
        print(
            tabulate(
                rows,
                headers=["Stage", "Scope", "Count", "Total", "Avg"],
                tablefmt="github"
            )
        )
    
    print("\n" + "=" * 65)
    print("                         SUMMARY")
    print("=" * 65)
    print(f"Output Markdown: {output_md_path.resolve()}")
    print(f"Figures folder : {figures_dir.resolve()}")
    print(f"Total Figures  : {len(figures)}")
    print(f"Placeholders   : {count_image_placeholders(final_md)} remaining")
    print(f"Total Time     : {time.time() - start_time:.2f} seconds")
    print("=" * 65)


if __name__ == "__main__":
    main()

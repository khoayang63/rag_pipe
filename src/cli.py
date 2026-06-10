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
from pipeline.parsers.doc_converter import PipelineConfig, convert_document, convert_documents, detect_input_format
from pipeline.parsers.figure_extractor import extract_figures
from pipeline.parsers.enrichment import enrich_markdown, count_image_placeholders, save_markdown
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Docling RAG Pipeline CLI — Run document conversion via Terminal."
    )
    
    ocr_choices = ["easyocr", "rapidocr", "tesseract", "tesseract_cli", "macocr", "kserve", "auto"]
    if sys.platform != "darwin" and "macocr" in ocr_choices:
        ocr_choices.remove("macocr")
    
    # Required/Optional arguments
    parser.add_argument(
        "input",
        nargs="*",
        type=str,
        help="Path(s) to input document(s). Multiple files supported. Optional if using --batch-ingest."
    )
    
    parser.add_argument(
        "--batch-ingest",
        action="store_true",
        dest="batch_ingest",
        help="Ingest all files in docs/incoming/ and move them to docs/processed/."
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
        choices=ocr_choices,
        default="rapidocr",
        help="OCR engine to use for standard pipeline (Default: rapidocr)."
    )
    parser.add_argument(
        "--layout-model",
        choices=["layout_v2", "heron", "heron_101", "egret_medium", "egret_large", "egret_xlarge"],
        default="heron",
        help="Layout analysis model to use for standard pipeline (Default: heron)."
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
    
    # End-to-End Ingestion options
    parser.add_argument(
        "--ingest",
        action="store_true",
        dest="ingest",
        help="Run chunking, generate embeddings and ingest into PostgreSQL pgvector."
    )
    parser.add_argument(
        "--chunk-method",
        choices=["hierarchical", "hybrid", "line_based"],
        default="hybrid",
        help="Chunking method to use for ingestion (Default: hybrid)."
    )
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk (Default: 512)."
    )
    parser.add_argument(
        "--no-chunk-merge",
        action="store_false",
        dest="chunk_merge_peers",
        help="Disable merging of peer elements in HybridChunker."
    )
    
    # Output path
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="cli_output",
        help="Directory to save outputs (Default: 'cli_output')."
    )
    
    return parser.parse_args()


def _process_single_file(input_path, args, config, gpu_info, out_dir, model_cache=None):
    """Process a single file and return (result, figures, enriched_md, output_md_path)."""
    start_time = time.time()
    
    print(f"\n[STEP 1/3] Converting document: {input_path.name}")
    if args.do_picture_description:
        print("[NOTE] Built-in Picture Description is ENABLED.")
    
    try:
        result = convert_document(str(input_path), config)
        elapsed = time.time() - start_time
        print(f"[SUCCESS] Conversion completed in {elapsed:.2f} seconds.")
        if config.ocr_engine == "auto":
            actual_ocr = result.pipeline_config.get("actual_ocr", "unknown")
            print(f"[INFO] Auto-detected OCR Engine used: {actual_ocr.upper()}")
    except Exception as e:
        print(f"\n[ERROR] Conversion failed for {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None, [], None, None
    
    # Prepare output directories
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract Figure Images
    figures = []
    if args.generate_picture_images:
        print(f"\n[STEP 2/3] Extracting figure images...")
        try:
            figures = extract_figures(result.document, str(figures_dir))
            print(f"[SUCCESS] Extracted {len(figures)} figure images into: {figures_dir}")
            
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
    
    # VLM Enrichment
    enriched_md = None
    if args.enrich and figures:
        if not gpu_info["available"]:
            print(f"\n[WARNING] CUDA is not available. Skipping downstream VLM enrichment.")
        else:
            print(f"\n[STEP 3/3] Running downstream figure description using {args.qwen_model}...")
            from pipeline.parsers.vlm_describer import describe_figures, load_model
            
            # Use cached model if available
            if model_cache and model_cache.get("model") is not None:
                model, processor = model_cache["model"], model_cache["processor"]
                print(f"Using cached model: {args.qwen_model}")
            else:
                print(f"Loading {args.qwen_model} model weights...")
                model, processor = load_model(model_id=args.qwen_model)
                if model_cache is not None:
                    model_cache["model"] = model
                    model_cache["processor"] = processor
            
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
    
    # Save output markdown
    final_md = enriched_md if enriched_md is not None else result.markdown
    output_md_path = out_dir / f"{input_path.stem}.md"
    save_markdown(final_md, str(output_md_path))
    
    # Database Ingestion (End-to-End Ingestion Pipeline)
    if args.ingest:
        print(f"\n[STEP 4/4] Chunking & Ingesting into pgvector...")
        try:
            # 1. Run Chunking
            from pipeline.processing.chunker import run_chunking
            print(f"Chunking document using '{args.chunk_method}' chunker (max_tokens={args.chunk_max_tokens}, merge_peers={args.chunk_merge_peers})...")
            chunk_result = run_chunking(
                document=result.document,
                method=args.chunk_method,
                max_tokens=args.chunk_max_tokens,
                merge_peers=args.chunk_merge_peers,
            )
            print(f"Generated {chunk_result.num_chunks} chunks.")
            
            if chunk_result.num_chunks > 0:
                # 2. Generate Embeddings using BAAI/bge-m3
                from pipeline.processing.embedder import get_embedder
                print("Loading BAAI/bge-m3 embedding model...")
                embedder = get_embedder()
                
                print("Generating dense embeddings (batch size = 16)...")
                payload_chunks = []
                texts_to_embed = []
                
                for chunk in chunk_result.chunks:
                    payload_chunks.append({
                        "index": chunk.index,
                        "text": chunk.text,
                        "contextualized": chunk.contextualized,
                        "page_no": chunk.page_no,
                        "chunk_type": chunk.chunk_type,
                        "headings": chunk.headings,
                        "captions": chunk.captions
                    })
                    texts_to_embed.append(chunk.contextualized)
                
                # Batch embed
                batch_size = 16
                all_embeddings = []
                for i in range(0, len(texts_to_embed), batch_size):
                    batch_texts = texts_to_embed[i:i+batch_size]
                    batch_embs = embedder.get_embeddings(batch_texts)
                    all_embeddings.extend(batch_embs)
                
                for idx, emb in enumerate(all_embeddings):
                    payload_chunks[idx]["embedding"] = emb
                
                # 3. Save to database
                from pipeline.indexer.postgres_store import VectorStore
                db = VectorStore()
                conn_info = db.test_connection()
                if not conn_info["connected"]:
                    print(f"[ERROR] Cannot connect to PostgreSQL: {conn_info['error']}")
                else:
                    doc_id = str(input_path.stem)  # Use filename stem as ID
                    print(f"Saving chunks to PostgreSQL database '{db.conn_params['database']}'...")
                    ingested_count = db.ingest_document(
                        doc_id=doc_id,
                        doc_name=input_path.name,
                        chunks=payload_chunks
                    )
                    print(f"[SUCCESS] Successfully ingested {ingested_count} chunks into pgvector!")
        except Exception as e:
            print(f"[ERROR] Chunking or DB Ingestion failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print timings
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
    
    return result, figures, enriched_md, output_md_path


def main():
    args = parse_args()
    
    # Check if batch-ingest is enabled
    if args.batch_ingest:
        incoming_dir = Path("docs/incoming")
        processed_dir = Path("docs/processed")
        incoming_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan files in docs/incoming
        supported_exts = {".pdf", ".docx", ".pptx", ".html", ".htm", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        files_to_process = []
        for item in incoming_dir.iterdir():
            if item.is_file() and item.suffix.lower() in supported_exts:
                files_to_process.append(item)
                
        if not files_to_process:
            print("\n[INFO] No files found to process in 'docs/incoming'.")
            # If the root of docs/ contains files, migrate them automatically
            root_docs = []
            for item in Path("docs").iterdir():
                if item.is_file() and item.suffix.lower() in supported_exts:
                    root_docs.append(item)
            if root_docs:
                print(f"[INFO] Found {len(root_docs)} files in the root of 'docs'. Moving them to 'docs/incoming' for batch processing...")
                for item in root_docs:
                    dest = incoming_dir / item.name
                    import shutil
                    try:
                        shutil.move(str(item), str(dest))
                        files_to_process.append(dest)
                    except Exception as e:
                        print(f"[ERROR] Could not move {item.name}: {e}")
            else:
                print("[INFO] Please place documents to ingest inside 'docs/incoming'.")
                sys.exit(0)
        
        input_paths = files_to_process
        args.ingest = True  # In batch mode, we force database ingestion
    else:
        if not args.input:
            print("[ERROR] Please specify input document(s) or use --batch-ingest.")
            sys.exit(1)
        input_paths = [Path(p) for p in args.input]
        for ip in input_paths:
            if not ip.exists():
                print(f"\n[ERROR] File does not exist: {ip}")
                sys.exit(1)
    
    is_batch = len(input_paths) > 1
    
    print("=" * 65)
    print("         DOCLING RAG PIPELINE — TERMINAL INFERENCE")
    print("=" * 65)
    if is_batch:
        print(f"Input files: {len(input_paths)} documents")
        for i, ip in enumerate(input_paths, 1):
            print(f"  [{i}] {ip.resolve()}")
    else:
        if input_paths:
            print(f"Input file : {input_paths[0].resolve()}")
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
    
    # 5. Process files
    global_start = time.time()
    out_dir_base = Path(args.output_dir)
    model_cache = {}  # Cache VLM model across files
    
    summary_rows = []
    
    for file_idx, input_path in enumerate(input_paths):
        if is_batch:
            print(f"\n{'━' * 65}")
            print(f"  FILE [{file_idx + 1}/{len(input_paths)}]: {input_path.name}")
            print(f"{'━' * 65}")
        
        # Each file gets its own dedicated subdirectory containing its md and figures
        out_dir = out_dir_base / input_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        
        result, figures, enriched_md, output_md_path = _process_single_file(
            input_path, args, config, gpu_info, out_dir, model_cache
        )
        
        if result is not None:
            final_md = enriched_md if enriched_md is not None else result.markdown
            summary_rows.append([
                input_path.name,
                f"{result.conversion_time:.2f}s",
                len(figures),
                count_image_placeholders(final_md),
                str(output_md_path),
            ])
            
            # If batch-ingest mode is active, move file from incoming to processed
            if args.batch_ingest:
                try:
                    processed_dir = Path("docs/processed")
                    processed_path = processed_dir / input_path.name
                    # Avoid file naming collision
                    if processed_path.exists():
                        stem, suffix = input_path.stem, input_path.suffix
                        processed_path = processed_dir / f"{stem}_{int(time.time())}{suffix}"
                    
                    import shutil
                    shutil.move(str(input_path), str(processed_path))
                    print(f"[SUCCESS] Moved processed file to: {processed_path}")
                except Exception as e:
                    print(f"[WARNING] Could not move processed file {input_path.name}: {e}")
    
    # 6. Final Summary
    total_elapsed = time.time() - global_start
    print("\n" + "=" * 65)
    print("                         SUMMARY")
    print("=" * 65)
    
    if is_batch:
        print(
            tabulate(
                summary_rows,
                headers=["File", "Time", "Figures", "Placeholders", "Output"],
                tablefmt="github"
            )
        )
        print(f"\nTotal files    : {len(summary_rows)} / {len(input_paths)}")
    else:
        if summary_rows:
            row = summary_rows[0]
            print(f"Output Markdown: {row[4]}")
            print(f"Figures folder : {out_dir / 'figures'}")
            print(f"Total Figures  : {row[2]}")
            print(f"Placeholders   : {row[3]} remaining")
    
    print(f"Total Time     : {total_elapsed:.2f} seconds")
    print("=" * 65)


if __name__ == "__main__":
    main()

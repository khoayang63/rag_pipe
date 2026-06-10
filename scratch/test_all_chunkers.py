import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.converter import PipelineConfig, create_standard_converter
from pipeline.chunker import run_all_chunking

config = PipelineConfig()
converter = create_standard_converter(config)
pdf_file = Path(__file__).resolve().parent.parent / "docs" / "de_toan5.pdf"

if not pdf_file.exists():
    print(f"File not found: {pdf_file}")
    sys.exit(1)

try:
    print("Converting PDF...")
    res = converter.convert(pdf_file)
    print("Running all 3 chunking methods...")
    results = run_all_chunking(res.document, max_tokens=256)
    
    for method, result in results.items():
        print(f"\n--- Method: {method} ({result.method_label}) ---")
        print(f"Num chunks: {result.num_chunks}")
        print(f"Total tokens: {result.total_tokens}")
        print(f"Avg tokens/chunk: {result.avg_tokens_per_chunk}")
        print(f"Chunking time: {result.chunking_time}s")
        if result.chunks:
            # Safely encode Unicode text for print
            safe_text = result.chunks[0].text[:80].encode('ascii', errors='replace').decode('ascii')
            safe_ctx = result.chunks[0].contextualized[:80].encode('ascii', errors='replace').decode('ascii')
            print(f"First chunk text (safe): {repr(safe_text)}")
            print(f"First chunk contextualized (safe): {repr(safe_ctx)}")
    print("\nAll chunking methods completed successfully!")
except Exception as e:
    import traceback
    print("Error during chunking:", e)
    traceback.print_exc()
    sys.exit(1)

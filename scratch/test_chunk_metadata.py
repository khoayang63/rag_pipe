import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.converter import PipelineConfig, create_standard_converter
from docling.chunking import HybridChunker

config = PipelineConfig()
converter = create_standard_converter(config)
pdf_file = Path(__file__).resolve().parent.parent / "docs" / "de_toan5.pdf"

try:
    res = converter.convert(pdf_file)
    chunker = HybridChunker()
    chunks = list(chunker.chunk(res.document))
    
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        c = chunks[0]
        print("Chunk class:", type(c).__name__)
        safe_text = c.text[:100].encode('ascii', errors='replace').decode('ascii')
        print("Text preview:", repr(safe_text))
        print("Meta export:", c.meta.export_json_dict())
        
        # Check doc_items
        if hasattr(c.meta, 'doc_items') and c.meta.doc_items:
            item = c.meta.doc_items[0]
            print("DocItem class:", type(item).__name__)
            print("DocItem dir:", [attr for attr in dir(item) if not attr.startswith("__")])
            if hasattr(item, 'prov') and item.prov:
                print("Prov type:", type(item.prov).__name__)
                print("Prov elements:")
                for p_idx, p in enumerate(item.prov):
                    print(f"  Prov[{p_idx}] class: {type(p).__name__}")
                    print(f"  Prov[{p_idx}] page_no: {p.page_no if hasattr(p, 'page_no') else 'N/A'}")
                    if hasattr(p, 'model_dump'):
                        print(f"  Prov[{p_idx}] details: {p.model_dump()}")
                    else:
                        print(f"  Prov[{p_idx}] dir: {dir(p)}")
except Exception as e:
    import traceback
    print("Error:", e)
    traceback.print_exc()

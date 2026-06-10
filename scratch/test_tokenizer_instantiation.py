from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker, HierarchicalChunker
from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker

try:
    # Test instantiating HuggingFaceTokenizer with from_pretrained and max_tokens
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name="BAAI/bge-m3",
        max_tokens=128
    )
    print("HuggingFaceTokenizer.from_pretrained: OK")
    print("Tokenizer max_tokens:", tokenizer.get_max_tokens())
    
    # Test HybridChunker
    chunker = HybridChunker(tokenizer=tokenizer)
    print("HybridChunker max_tokens:", chunker.max_tokens)
    
    # Test LineBasedTokenChunker
    line_chunker = LineBasedTokenChunker(tokenizer=tokenizer)
    print("LineBasedTokenChunker max_tokens:", line_chunker.max_tokens)
except Exception as e:
    import traceback
    print("Error:", e)
    traceback.print_exc()

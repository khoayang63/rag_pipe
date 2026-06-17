import sys
from transformers import AutoTokenizer, AutoModel, AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq

def download_embedding():
    print("\n==================================================")
    print("Downloading BAAI/bge-m3 (required for Vector DB)...")
    print("Size: ~2.2 GB")
    print("==================================================")
    AutoTokenizer.from_pretrained("BAAI/bge-m3")
    AutoModel.from_pretrained("BAAI/bge-m3")
    print("BAAI/bge-m3 downloaded successfully.")

def download_vlm(model_id):
    print("\n==================================================")
    print(f"Downloading VLM: {model_id}...")
    print("Size: ~1 GB to 4.5 GB depending on model")
    print("==================================================")
    # Use trust_remote_code=True for Qwen-VL models
    AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True)
    print(f"{model_id} downloaded successfully.")

def download_reranker():
    print("\n==================================================")
    print("Downloading BAAI/bge-reranker-v2-m3 (Reranking)...")
    print("Size: ~2.2 GB")
    print("==================================================")
    from transformers import AutoModelForSequenceClassification
    AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
    print("BAAI/bge-reranker-v2-m3 downloaded successfully.")

def download_spell_corrector():
    print("\n==================================================")
    print("Downloading bmd1905/vietnamese-correction-v2 (Spell Correction)...")
    print("Size: ~1.5 GB")
    print("==================================================")
    from transformers import AutoModelForSeq2SeqLM
    AutoTokenizer.from_pretrained("bmd1905/vietnamese-correction-v2")
    AutoModelForSeq2SeqLM.from_pretrained("bmd1905/vietnamese-correction-v2")
    print("bmd1905/vietnamese-correction-v2 downloaded successfully.")

def main():
    print("==================================================")
    print("   Docling RAG Pipeline Model Downloader")
    print("==================================================")
    print("Use this script to pre-download model weights to prevent")
    print("Streamlit interface freezes on first run.")
    print("==================================================\n")

    print("Please select what you would like to download:")
    print("1. [Recommended] Default Setup (BAAI/bge-m3 + Qwen3-VL-2B-Instruct)")
    print("2. BAAI/bge-m3 (2.2 GB) - Required for Vector Search & DB")
    print("3. BAAI/bge-reranker-v2-m3 (2.2 GB) - Required for Reranking")
    print("4. Qwen/Qwen3-VL-2B-Instruct (4.5 GB) - Default recommended VLM")
    print("5. Qwen/Qwen2-VL-2B-Instruct (4.5 GB) - Alternate VLM")
    print("6. HuggingFaceTB/SmolVLM-256M-Instruct (0.8 GB) - Ultra-light VLM")
    print("7. bmd1905/vietnamese-correction-v2 (~1.5 GB) - Vietnamese Spell Correction")
    print("8. Download All (All of the above)")
    print("9. Exit")
    
    try:
        choice = input("\nEnter choice (1-9): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)
        
    if choice == "1":
        download_embedding()
        download_vlm("Qwen/Qwen3-VL-2B-Instruct")
    elif choice == "2":
        download_embedding()
    elif choice == "3":
        download_reranker()
    elif choice == "4":
        download_vlm("Qwen/Qwen3-VL-2B-Instruct")
    elif choice == "5":
        download_vlm("Qwen/Qwen2-VL-2B-Instruct")
    elif choice == "6":
        download_vlm("HuggingFaceTB/SmolVLM-256M-Instruct")
    elif choice == "7":
        download_spell_corrector()
    elif choice == "8":
        download_embedding()
        download_reranker()
        download_spell_corrector()
        download_vlm("Qwen/Qwen3-VL-2B-Instruct")
        download_vlm("Qwen/Qwen2-VL-2B-Instruct")
        download_vlm("HuggingFaceTB/SmolVLM-256M-Instruct")
    else:
        print("Exiting.")
        sys.exit(0)
        
    print("\n--------------------------------------------------")
    print("All selected models downloaded successfully!")
    print("You can now safely run: streamlit run src/app.py")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()

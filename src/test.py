import os
import torch
import sys
from dotenv import load_dotenv

load_dotenv()

print("HF_TOKEN:", os.environ.get("HF_TOKEN"))
print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

# Tạm thời bật lại cảnh báo tiến trình download của transformers trong script test này
if "TRANSFORMERS_VERBOSITY" in os.environ:
    del os.environ["TRANSFORMERS_VERBOSITY"]

print("\n--- Choose HuggingFace Model to Test Load ---")
print("1. Qwen/Qwen3-VL-2B-Instruct (2B, ~4.5GB) - Figure Describer Model (Default)")
print("2. ibm-granite/granite-docling-258M (258M, ~500MB) - Docling VLM Model")
print("3. Qwen/Qwen2-VL-2B-Instruct (2B, ~4.5GB) - Figure Describer Model (Lighter)")
print("4. HuggingFaceTB/SmolVLM-256M-Instruct (256M, ~500MB) - Figure Describer Model (Super Light)")

choice = "1"
if sys.stdin.isatty():
    try:
        choice = input("\nEnter your choice (1, 2, 3 or 4, default is 1): ").strip()
        if not choice:
            choice = "1"
    except Exception:
        choice = "1"
else:
    # Check command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ["1", "2", "3", "4"]:
        choice = sys.argv[1]

# Set model details based on choice
if choice == "2":
    MODEL_ID = "ibm-granite/granite-docling-258M"
    model_type = "IBM Granite VLM"
    use_auto_model = True
elif choice == "3":
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    model_type = "Qwen2-VL 2B"
    use_auto_model = False
elif choice == "4":
    MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
    model_type = "SmolVLM 256M"
    use_auto_model = False
else:
    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    model_type = "Qwen3-VL 2B"
    use_auto_model = False

print(f"\nTesting load for {model_type} ({MODEL_ID})...")
print("Note: If running for the first time, this will download model weights from HuggingFace.")
print("The terminal will display progress bars (tqdm) for the downloads.\n")

if torch.cuda.is_available():
    try:
        from transformers import AutoProcessor, AutoModel
        try:
            from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForVision2Seq
        
        # Tự động chọn kiểu dữ liệu 16-bit tối ưu để tiết kiệm VRAM
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using dtype: {torch_dtype} (Optimized 16-bit)")
        
        print(f"1. Loading model weights from '{MODEL_ID}' (device_map='auto')...")
        if use_auto_model:
            model = AutoModel.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                device_map="auto",
            )
        else:
            model = AutoModelForVision2Seq.from_pretrained(
                MODEL_ID,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        print("Model loaded successfully!")
        
        print(f"2. Loading processor...")
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("Processor loaded successfully!")
        
        print(f"\n[SUCCESS] Both model and processor are fully loaded and cached on your GPU!")
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
else:
    # CPU fallback for lightweight models
    if choice in ["2", "4"]:
        try:
            print(f"\n[INFO] CUDA is not available. Loading {model_type} on CPU...")
            from transformers import AutoProcessor, AutoModel
            try:
                from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
            except ImportError:
                from transformers import AutoModelForVision2Seq
            if use_auto_model:
                model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
            else:
                model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            print(f"\n[SUCCESS] Both model and processor are fully loaded on CPU!")
        except Exception as e:
            print(f"\n[ERROR] Failed to load model on CPU: {e}")
    else:
        print(f"\n[WARNING] CUDA is not available. {model_type} cannot be loaded on CPU in this setup.")
"""
Configuration module for the Docling RAG Pipeline.

Loads environment variables, sets up output directories,
and handles HuggingFace authentication.
"""

import os
import sys

# Tắt các thông báo cảnh báo và lỗi advisory phiền phức từ thư viện transformers trước khi import
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import tempfile
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Tắt các cảnh báo (Warnings) từ thư viện bên thứ ba để giữ sạch Terminal
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import transformers
    # Chỉ hiện lỗi nghiêm trọng từ transformers logger, ẩn các cảnh báo thông thường
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

# Load environment variables from .env file
load_dotenv()


def get_hf_token() -> str:
    """Retrieve HuggingFace token from environment."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set.\n"
            "Please set it before running:\n"
            "  Windows:  set HF_TOKEN=hf_your_token_here\n"
            "  Linux:    export HF_TOKEN=hf_your_token_here"
        )
    return token


def login_huggingface(token: str) -> bool:
    """Login to HuggingFace Hub with the provided token."""
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        return True
    except Exception as e:
        print(f"HuggingFace login failed: {e}", file=sys.stderr)
        return False


def get_output_dir() -> Path:
    """Get or create the output directory for processed files."""
    output_dir = Path(tempfile.gettempdir()) / "docling_rag_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_figures_dir(session_id: str) -> Path:
    """Get or create the figures directory for a specific session."""
    figures_dir = get_output_dir() / session_id / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def get_markdown_dir(session_id: str) -> Path:
    """Get or create the markdown output directory for a specific session."""
    md_dir = get_output_dir() / session_id / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)
    return md_dir


def check_gpu_available() -> dict:
    """Check CUDA GPU availability and return device info."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_total / (1024 ** 3)
            return {
                "available": True,
                "device": device_name,
                "vram_gb": round(vram_gb, 1),
                "torch_version": torch.__version__,
            }
    except ImportError:
        pass
    return {
        "available": False,
        "device": "CPU only",
        "vram_gb": 0,
        "torch_version": "N/A",
    }


# Supported file extensions mapped to display names
SUPPORTED_FORMATS = {
    ".pdf": "PDF Document",
    ".docx": "Word Document",
    ".pptx": "PowerPoint Presentation",
    ".html": "HTML Document",
    ".htm": "HTML Document",
    ".png": "PNG Image",
    ".jpg": "JPEG Image",
    ".jpeg": "JPEG Image",
    ".tiff": "TIFF Image",
    ".tif": "TIFF Image",
    ".bmp": "BMP Image",
}

ACCEPTED_EXTENSIONS = list(SUPPORTED_FORMATS.keys())

"""
Multimodal embedding generation module using BAAI/BGE-VL-large.
Loads the model using Hugging Face transformers library directly with trust_remote_code=True.
"""

import torch
from transformers import AutoModel
from PIL import Image
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class BgeVisualizedEmbedder:
    """Class to compute multimodal embeddings using BAAI/BGE-VL-large."""

    def __init__(self, model_name: str = "BAAI/BGE-VL-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[VisualEmbedder] Initializing {model_name} on device: {self.device}...")
        logger.info("[VisualEmbedder] Loading model. Downloading if not cached...")
        # Choose torch_dtype to ensure compatibility and avoid BFloat16 errors.
        # BAAI/BGE-VL-large defaults to bfloat16, but some systems do not support bfloat16 vision operators.
        self.torch_dtype = torch.float32
        
        # Load the model with trust_remote_code=True
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        logger.info("[VisualEmbedder] Setting processor...")
        self.model.set_processor(model_name)
        self.model.eval()
        
        logger.info("[VisualEmbedder] Multimodal model loaded and ready.")

    def embed_text(self, text: str) -> List[float]:
        """Compute embedding for a text query."""
        if not text:
            return []
            
        with torch.no_grad():
            emb = self.model.encode(text=text)
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            
            # Convert to list
            emb_list = emb.tolist()
            # If batch output format, return the first element
            if isinstance(emb_list[0], list):
                return emb_list[0]
            return emb_list

    def embed_image(self, image_path: str) -> List[float]:
        """Compute embedding for a document figure image."""
        if not image_path:
            return []
            
        with torch.no_grad():
            # The model's encode method accepts image paths or PIL Images
            emb = self.model.encode(images=image_path)
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
                
            # Convert to list
            emb_list = emb.tolist()
            # If batch output format, return the first element
            if isinstance(emb_list[0], list):
                return emb_list[0]
            return emb_list


# Lazy-loaded singleton instance
_visual_embedder_instance = None

def get_visual_embedder() -> BgeVisualizedEmbedder:
    """Retrieve or initialize the global BgeVisualizedEmbedder instance."""
    global _visual_embedder_instance
    if _visual_embedder_instance is None:
        _visual_embedder_instance = BgeVisualizedEmbedder()
    return _visual_embedder_instance

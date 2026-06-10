"""
Embedding generation module using BAAI/bge-m3 dense embeddings.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List

class BgeEmbedder:
    """Class to compute dense embeddings for text chunks using BAAI/bge-m3."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[Embedder] Initializing {model_name} on device: {self.device}...")
        print("[Embedder] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("[Embedder] Loading model weights (~2.2 GB). Downloading if not cached...")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[Embedder] Model loaded and ready.\n")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute 1024-dimensional normalized dense embeddings for a list of texts."""
        if not texts:
            return []
            
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # CLS pooling
            embeddings = outputs.last_hidden_state[:, 0]
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings.cpu().numpy().tolist()


# Lazy-loaded singleton instance
_embedder_instance = None

def get_embedder() -> BgeEmbedder:
    """Retrieve or initialize the global BgeEmbedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = BgeEmbedder()
    return _embedder_instance

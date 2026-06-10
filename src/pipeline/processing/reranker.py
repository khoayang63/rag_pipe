"""
Reranking module using BAAI/bge-reranker-v2-m3.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

class BgeReranker:
    """Class to compute relevance scores for query-passage pairs using BAAI/bge-reranker-v2-m3."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[Reranker] Initializing {model_name} on device: {self.device}...")
        print("[Reranker] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("[Reranker] Loading model weights (~2.2 GB). Downloading if not cached...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[Reranker] Model loaded and ready.\n")

    def compute_scores(self, query: str, passages: List[str]) -> List[float]:
        """Compute similarity/relevance scores for a query and a list of passages."""
        if not passages:
            return []
            
        pairs = [[query, passage] for passage in passages]
        
        # Process in batches to prevent GPU out-of-memory errors
        batch_size = 16
        scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits.view(-1,)
                # Sigmoid to normalize logit scores to [0, 1] range for user friendly display
                probs = torch.sigmoid(logits)
                scores.extend(probs.cpu().numpy().tolist())
                
        return scores


# Lazy-loaded singleton instance
_reranker_instance = None

def get_reranker() -> BgeReranker:
    """Retrieve or initialize the global BgeReranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BgeReranker()
    return _reranker_instance

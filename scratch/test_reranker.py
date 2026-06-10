import os
import sys

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.processing.reranker import get_reranker

def main():
    print("Testing BgeReranker singleton and compute_scores function...")
    try:
        reranker = get_reranker()
    except Exception as e:
        print(f"Failed to load reranker: {e}")
        sys.exit(1)
        
    query = "What is the capital of France?"
    passages = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany and its largest city.",
        "The attention mechanism is a deep learning technique that allows models to focus on specific parts of the input.",
        "France has a population of over 67 million people."
    ]
    
    print(f"\nQuery: '{query}'")
    print("\nPassages:")
    for idx, passage in enumerate(passages):
        print(f"  {idx + 1}. {passage}")
        
    print("\nComputing scores...")
    try:
        scores = reranker.compute_scores(query, passages)
        print("\nScores:")
        for idx, (passage, score) in enumerate(zip(passages, scores)):
            print(f"  Passage {idx + 1} Score: {score:.4f}")
        print("\nTest passed successfully!")
    except Exception as e:
        print(f"Failed to compute scores: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

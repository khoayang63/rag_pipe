"""
Vietnamese spell correction module using bmd1905/vietnamese-correction-v2.

Fine-tuned BARTpho Seq2Seq model for Vietnamese error correction.
Computes word-level diffs to highlight spelling and layout changes.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

MODEL_ID = "bmd1905/vietnamese-correction-v2"


@dataclass
class CorrectionDiff:
    """A single diff between original and corrected text within a chunk."""
    original: str
    corrected: str
    tag: str  # "replace", "insert", "delete"


@dataclass
class ParagraphCorrection:
    """Correction result for a single chunk."""
    index: int
    original: str
    corrected: str
    diffs: list[CorrectionDiff] = field(default_factory=list)
    has_changes: bool = False
    is_skipped: bool = False  # True for syntax/non-alphabetic chunks


def should_skip_chunk(text: str) -> bool:
    """Check if a chunk should be skipped (no letters, too short, email/phone labels)."""
    if not text or not text.strip():
        return True
    
    # 1. Skip if there are no alphabetic characters at all (only numbers, punctuation, spaces)
    if not any(c.isalpha() for c in text):
        return True
    
    # 2. Skip if the text content is extremely short and contains label indicators (e.g. Phone:, Email:)
    clean_text = re.sub(r"\s+", "", text)
    if len(clean_text) < 12 and (":" in text or "@" in text):
        return True
        
    return False


def _compute_diffs(original: str, corrected: str) -> list[CorrectionDiff]:
    """Compute word-level diffs between original and corrected text."""
    if original == corrected:
        return []

    diffs = []
    matcher = difflib.SequenceMatcher(None, original, corrected)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        diffs.append(CorrectionDiff(
            original=original[i1:i2],
            corrected=corrected[j1:j2],
            tag=tag,
        ))

    return diffs


class SpellCorrector:
    """Vietnamese spell corrector using bmd1905/vietnamese-correction-v2."""

    def __init__(self, model_id: str = MODEL_ID):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[SpellCorrector] Initializing {model_id} on device: {self.device}...")
        print("[SpellCorrector] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("[SpellCorrector] Loading model weights. Downloading if not cached...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print("[SpellCorrector] Model loaded and ready.\n")

    def correct_text(self, text: str, max_length: int = 512) -> str:
        """Correct a single text string."""
        import torch

        if not text or not text.strip():
            return text

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def correct_paragraph(self, text: str, index: int) -> ParagraphCorrection:
        """Correct a single text chunk and compute diffs."""
        # Check if we should skip
        if should_skip_chunk(text):
            return ParagraphCorrection(
                index=index,
                original=text,
                corrected=text,
                diffs=[],
                has_changes=False,
                is_skipped=True,
            )

        corrected = self.correct_text(text)
        diffs = _compute_diffs(text, corrected)

        return ParagraphCorrection(
            index=index,
            original=text,
            corrected=corrected,
            diffs=diffs,
            has_changes=len(diffs) > 0,
            is_skipped=False,
        )


# ──────────────────────────────────────────────
# Lazy-loaded singleton
# ──────────────────────────────────────────────
_corrector_instance: Optional[SpellCorrector] = None


def get_spell_corrector() -> SpellCorrector:
    """Retrieve or initialize the global SpellCorrector instance."""
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = SpellCorrector()
    return _corrector_instance

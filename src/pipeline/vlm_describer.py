"""
VLM Figure Describer module.

Uses Qwen3-VL-2B-Instruct to generate natural language descriptions
of extracted figures. Per Docling Skill, Qwen3-VL is preferred
over Docling's built-in picture description models for RAG workflows.
"""

import gc
import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class DescriptionResult:
    """Result of VLM figure description."""

    descriptions: list[str]
    model_name: str
    inference_time: float
    gpu_used: bool


SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in describing figures, charts, "
    "diagrams, and visual content from academic and technical documents. "
    "Provide clear, detailed descriptions that capture the key information "
    "conveyed by each figure."
)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def check_gpu() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_model(model_id: str = MODEL_ID):
    """
    Load the vision-language model and processor using AutoModelForVision2Seq.

    Returns:
        Tuple of (model, processor) or (None, None) if GPU not available.
    """
    if not check_gpu():
        return None, None

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForVision2Seq
    import torch

    # Sử dụng kiểu dữ liệu 16-bit tối ưu để tiết kiệm VRAM và tránh offload sang CPU
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    return model, processor


def describe_figures(
    figures: list,
    model=None,
    processor=None,
    prompt: str = "Describe this figure in detail.",
    model_id: str = MODEL_ID,
) -> DescriptionResult:
    """
    Generate descriptions for a batch of figures using a Vision-Language Model.

    Per Docling Skill RAG workflow:
    PDF -> Docling Figure Extraction -> Export Figure Images
    -> VLM -> Figure Description -> Markdown Enrichment

    Args:
        figures: List of FigureData objects with pil_image attribute
        model: Preloaded VLM model (or None to load)
        processor: Preloaded processor (or None to load)
        prompt: The text prompt for figure description
        model_id: The HuggingFace model ID to load if not preloaded

    Returns:
        DescriptionResult with list of descriptions
    """
    if not figures:
        return DescriptionResult(
            descriptions=[],
            model_name=model_id,
            inference_time=0.0,
            gpu_used=False,
        )

    # Load model if not provided
    if model is None or processor is None:
        if not check_gpu():
            return DescriptionResult(
                descriptions=["(GPU not available — skipped)"] * len(figures),
                model_name=model_id,
                inference_time=0.0,
                gpu_used=False,
            )
        model, processor = load_model(model_id)

    start_time = time.time()

    # Build batch messages
    batch_messages = []
    batch_images = [fig.pil_image for fig in figures]

    for image in batch_images:
        batch_messages.append(
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        )

    # Tokenize
    texts = [
        processor.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
        )
        for msg in batch_messages
    ]

    processor.tokenizer.padding_side = "left"

    inputs = processor(
        text=texts,
        images=batch_images,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # Generate
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    inference_time = time.time() - start_time

    # Cleanup
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    return DescriptionResult(
        descriptions=output_text,
        model_name=model_id,
        inference_time=inference_time,
        gpu_used=True,
    )

"""
Figure Extractor module.

Extracts figures (PictureItem) from a converted Docling document,
saves them as images, and returns metadata for each figure.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class FigureData:
    """Metadata and image data for an extracted figure."""

    index: int
    image_path: str
    caption: Optional[str]
    picture_ref: str
    page_no: int
    bbox: object  # BoundingBox from docling
    pil_image: object  # PIL Image
    classification: Optional[str] = None
    confidence: Optional[float] = None


def extract_figures(document, output_dir: str) -> list[FigureData]:
    """
    Extract all figures from a Docling document and save as PNG files.

    Per Docling Skill: iterate doc.pictures, use pic.get_image(doc),
    and collect captions from doc.texts.

    Args:
        document: Docling Document object (result.document)
        output_dir: Directory to save extracted figure images

    Returns:
        List of FigureData with metadata for each figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build text reference map for captions
    text_map = {t.self_ref: t for t in document.texts}

    figures = []

    for i, pic in enumerate(document.pictures):
        # Get the PIL image
        image = pic.get_image(document)
        if image is None:
            continue

        # Save the image
        img_filename = f"figure_{i}.png"
        img_path = output_path / img_filename
        image.save(str(img_path))

        # Extract caption if available
        caption = None
        if pic.captions:
            caption_ref = pic.captions[0].cref
            if caption_ref in text_map:
                caption = text_map[caption_ref].text

        # Get provenance info (page number, bounding box)
        page_no = 0
        bbox = None
        if pic.prov:
            prov = pic.prov[0]
            page_no = prov.page_no
            bbox = prov.bbox

        # Extract classification if available
        classification = None
        confidence = None
        if hasattr(pic, "annotations") and pic.annotations:
            cls_data = pic.annotations[0]
            if hasattr(cls_data, "predicted_classes") and cls_data.predicted_classes:
                top = cls_data.predicted_classes[0]
                classification = top.class_name
                confidence = top.confidence

        figures.append(
            FigureData(
                index=i,
                image_path=str(img_path),
                caption=caption,
                picture_ref=pic.self_ref,
                page_no=page_no,
                bbox=bbox,
                pil_image=image,
                classification=classification,
                confidence=confidence,
            )
        )

    return figures

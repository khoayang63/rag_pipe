"""
Unit tests for OCR language mapping and compatibility filtering.
"""

import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pipeline.parsers.doc_converter import _map_ocr_languages, _filter_easyocr_compatibility


def test_map_ocr_languages_easyocr():
    # zh defaults to ch_sim in EasyOCR
    langs = ["en", "zh"]
    res = _map_ocr_languages(langs, "easyocr")
    assert res == ["en", "ch_sim"]

    # Incompatible script filtering (first non-en decides the category)
    # 1. "vi" is Latin, so "zh" (ch_sim) is discarded
    langs2 = ["en", "vi", "zh"]
    res2 = _map_ocr_languages(langs2, "easyocr")
    assert res2 == ["en", "vi"]

    # 2. "zh" (ch_sim) is Chinese, so "vi" (Latin) is discarded
    langs3 = ["zh", "vi"]
    res3 = _map_ocr_languages(langs3, "easyocr")
    assert res3 == ["ch_sim"]

    # 3. Cyrillic "ru" and Arabic "ar" are incompatible. "ru" is first, so "ar" is discarded.
    langs4 = ["ru", "ar"]
    res4 = _map_ocr_languages(langs4, "easyocr")
    assert res4 == ["ru"]

    # Duplicates are removed while preserving order
    langs5 = ["en", "zh", "en", "zh"]
    res5 = _map_ocr_languages(langs5, "easyocr")
    assert res5 == ["en", "ch_sim"]


def test_map_ocr_languages_tesseract():
    langs = ["en", "vi", "zh", "fr"]
    res = _map_ocr_languages(langs, "tesseract")
    assert res == ["eng", "vie", "chi_sim", "fra"]


def test_map_ocr_languages_rapidocr():
    langs = ["en", "vi", "zh"]
    res = _map_ocr_languages(langs, "rapidocr")
    assert res == ["english", "vi", "chinese"]


def test_map_ocr_languages_macocr():
    langs = ["en", "vi", "zh", "ja"]
    res = _map_ocr_languages(langs, "macocr")
    assert res == ["en-US", "vi-VN", "zh-CN", "ja-JP"]


def test_map_ocr_languages_unknown_or_default():
    langs = ["en", "vi"]
    res = _map_ocr_languages(langs, "unknown_engine")
    assert res == ["en", "vi"]

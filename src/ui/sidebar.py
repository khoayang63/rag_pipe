"""
Sidebar component for pipeline configuration.

Provides toggles for all pipeline options, pipeline mode selector,
and displays current configuration as formatted JSON.
"""

import streamlit as st
from pipeline.converter import PipelineConfig


def render_sidebar() -> PipelineConfig:
    """
    Render the sidebar with pipeline configuration options.

    Returns:
        PipelineConfig with user-selected options
    """
    with st.sidebar:
        # Sidebar header
        st.markdown(
            """
            <div style="margin-bottom: 1.5rem;">
                <h2 style="margin:0; font-size:1.2rem;">
                    <span style="color: #34d399;">&#9670;</span> Pipeline Config
                </h2>
                <p style="margin:4px 0 0 0; font-size:0.78rem; color:#5c5c6e;">
                    Configure the document processing pipeline
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Pipeline mode
        pipeline_mode = st.radio(
            "Pipeline Mode",
            options=["standard", "vlm"],
            format_func=lambda x: {
                "standard": "Standard (OCR + Layout)",
                "vlm": "VLM (Vision Language Model)",
            }[x],
            help=(
                "**Standard**: Uses StandardPdfPipeline with OCR, layout analysis, "
                "table/formula/figure extraction. Best for RAG.\n\n"
                "**VLM**: Uses VlmPipeline for end-to-end vision-language model "
                "conversion. Best for image-heavy documents."
            ),
        )

        st.divider()

        # Downstream VLM Describer selection
        st.markdown(
            '<p style="font-size:0.8rem; color:#9898a6; font-weight:600; '
            'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">'
            "Figure Describer VLM</p>",
            unsafe_allow_html=True,
        )
        downstream_model = st.selectbox(
            "Describer Model",
            options=[
                "Qwen/Qwen3-VL-2B-Instruct",
                "Qwen/Qwen2-VL-2B-Instruct",
                "HuggingFaceTB/SmolVLM-256M-Instruct"
            ],
            index=0,
            help="Vision-Language Model to describe figures downstream when clicking 'Describe Figures'."
        )

        st.divider()

        # Standard pipeline options
        if pipeline_mode == "standard":
            st.markdown(
                '<p style="font-size:0.8rem; color:#9898a6; font-weight:600; '
                'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">'
                "Processing Options</p>",
                unsafe_allow_html=True,
            )

            do_ocr = st.toggle("OCR", value=True, help="Optical Character Recognition for scanned documents")
            do_table = st.toggle("Table Structure", value=True, help="Extract and structure tables")
            do_formula = st.toggle("Formula Enrichment", value=True, help="Detect and extract mathematical formulas")
            do_code = st.toggle("Code Enrichment", value=True, help="Detect and extract code blocks")
            do_pic_desc = st.toggle(
                "Picture Description (built-in)",
                value=False,
                help="Use Docling's built-in picture description. Disable if using Qwen3-VL instead.",
            )
            do_pic_class = st.toggle(
                "Picture Classification",
                value=False,
                help="Classify extracted pictures/figures (e.g. chart, diagram, photo).",
            )

            st.markdown(
                '<p style="font-size:0.8rem; color:#9898a6; font-weight:600; '
                'text-transform:uppercase; letter-spacing:0.05em; margin:1rem 0 0.5rem;">'
                "Image Generation</p>",
                unsafe_allow_html=True,
            )

            gen_page = st.toggle("Generate Page Images", value=False, help="Generate images for each page")
            gen_pic = st.toggle("Generate Picture Images", value=True, help="Generate cropped images for each figure")

            st.divider()

            st.markdown(
                '<p style="font-size:0.8rem; color:#9898a6; font-weight:600; '
                'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">'
                "Advanced</p>",
                unsafe_allow_html=True,
            )

            images_scale = st.slider(
                "Image Scale",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.25,
                help="1.0 for standard PDFs. 2.0 for scanned/tiny text. Higher = better OCR but more RAM/VRAM.",
            )

            ocr_langs = st.multiselect(
                "OCR Languages",
                options=["en", "vi", "zh", "ja", "ko", "fr", "de", "es", "pt", "it", "ru", "ar"],
                default=["en"],
                help="Select OCR languages for text recognition",
            )

            config = PipelineConfig(
                pipeline_mode="standard",
                do_ocr=do_ocr,
                do_table_structure=do_table,
                do_formula_enrichment=do_formula,
                do_code_enrichment=do_code,
                do_picture_description=do_pic_desc,
                do_picture_classification=do_pic_class,
                generate_page_images=gen_page,
                generate_picture_images=gen_pic,
                images_scale=images_scale,
                ocr_languages=ocr_langs,
                downstream_model=downstream_model,
            )

        else:
            # VLM pipeline options
            st.markdown(
                '<p style="font-size:0.8rem; color:#9898a6; font-weight:600; '
                'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">'
                "VLM Model Preset</p>",
                unsafe_allow_html=True,
            )

            vlm_preset = st.selectbox(
                "Model Preset",
                options=[
                    "GRANITE_VISION_TRANSFORMERS",
                    "GRANITEDOCLING_TRANSFORMERS",
                    "SMOLDOCLING_TRANSFORMERS",
                ],
                help="Select the VLM model preset from docling.datamodel.vlm_model_specs",
            )

            st.divider()

            st.markdown(
                '<p style="font-size:0.8rem; color:#9898a6; font-weight:600; '
                'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">'
                "VLM Parameters</p>",
                unsafe_allow_html=True,
            )

            vlm_scale = st.slider("Scale", min_value=0.5, max_value=2.0, value=1.0, step=0.25)
            vlm_max_size = st.number_input("Max Size (px)", min_value=256, max_value=2048, value=1024, step=128)
            vlm_max_tokens = st.number_input("Max New Tokens", min_value=64, max_value=2048, value=512, step=64)

            images_scale = st.slider(
                "Image Scale",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.25,
                help="Scale factor for pipeline images",
            )

            config = PipelineConfig(
                pipeline_mode="vlm",
                vlm_preset=vlm_preset,
                vlm_scale=vlm_scale,
                vlm_max_size=vlm_max_size,
                vlm_max_new_tokens=vlm_max_tokens,
                images_scale=images_scale,
                downstream_model=downstream_model,
            )

        # Config JSON preview
        st.divider()

        with st.expander("View Config JSON", expanded=False):
            import json

            if config.pipeline_mode == "standard":
                display_config = {
                    "pipeline": "StandardPdfPipeline",
                    "do_ocr": config.do_ocr,
                    "do_table_structure": config.do_table_structure,
                    "do_formula_enrichment": config.do_formula_enrichment,
                    "do_code_enrichment": config.do_code_enrichment,
                    "do_picture_description": config.do_picture_description,
                    "do_picture_classification": config.do_picture_classification,
                    "generate_page_images": config.generate_page_images,
                    "generate_picture_images": config.generate_picture_images,
                    "images_scale": config.images_scale,
                    "ocr_languages": config.ocr_languages,
                    "downstream_model": config.downstream_model,
                }
            else:
                display_config = {
                    "pipeline": "VlmPipeline",
                    "vlm_preset": config.vlm_preset,
                    "vlm_scale": config.vlm_scale,
                    "vlm_max_size": config.vlm_max_size,
                    "vlm_max_new_tokens": config.vlm_max_new_tokens,
                    "images_scale": config.images_scale,
                    "downstream_model": config.downstream_model,
                }

            st.code(json.dumps(display_config, indent=2), language="json")

    return config

"""
Docling RAG Pipeline — Streamlit Application

Main entry point for the document processing pipeline with web UI.
Supports PDF, DOCX, PPTX, HTML, and image files.

Run: streamlit run src/app.py
"""

import sys
import os
import uuid
import streamlit as st

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_hf_token, login_huggingface, get_figures_dir, check_gpu_available
from ui.styles import inject_custom_css
from ui.sidebar import render_sidebar
from ui.upload import render_upload
from ui.markdown_viewer import render_markdown_viewer
from ui.figure_gallery import render_figure_gallery
from ui.pipeline_info import render_pipeline_info
from pipeline.converter import convert_document
from pipeline.figure_extractor import extract_figures
from pipeline.enrichment import enrich_markdown, count_image_placeholders


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Docling RAG Pipeline",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Docling RAG Pipeline — Document processing with Docling + Qwen3-VL",
    },
)

# ──────────────────────────────────────────────
# Custom Styles
# ──────────────────────────────────────────────
inject_custom_css()

# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "conversion_result" not in st.session_state:
    st.session_state.conversion_result = None

if "figures" not in st.session_state:
    st.session_state.figures = None

if "descriptions" not in st.session_state:
    st.session_state.descriptions = None

if "description_result" not in st.session_state:
    st.session_state.description_result = None

if "enriched_md" not in st.session_state:
    st.session_state.enriched_md = None

if "hf_logged_in" not in st.session_state:
    st.session_state.hf_logged_in = False

if "vlm_model" not in st.session_state:
    st.session_state.vlm_model = None

if "vlm_processor" not in st.session_state:
    st.session_state.vlm_processor = None

# ──────────────────────────────────────────────
# HuggingFace Authentication
# ──────────────────────────────────────────────
if not st.session_state.hf_logged_in:
    try:
        token = get_hf_token()
        if login_huggingface(token):
            st.session_state.hf_logged_in = True
    except EnvironmentError as e:
        st.sidebar.error(str(e))

# ──────────────────────────────────────────────
# Sidebar — Pipeline Config
# ──────────────────────────────────────────────
config = render_sidebar()

# ──────────────────────────────────────────────
# Main Content Area
# ──────────────────────────────────────────────

# App Header
st.markdown(
    """
    <div class="app-header">
        <div class="logo-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
                 stroke="#0c0c0f" stroke-width="2" stroke-linecap="round"
                 stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
                <polyline points="10 9 9 9 8 9"/>
            </svg>
        </div>
        <div class="header-text">
            <h1>Docling RAG Pipeline</h1>
            <p>Document conversion, figure extraction, and VLM enrichment</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Step 1: File Upload
# ──────────────────────────────────────────────
file_path = render_upload()

# ──────────────────────────────────────────────
# Step 2: Convert Button
# ──────────────────────────────────────────────
if file_path:
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        convert_clicked = st.button(
            "Convert Document",
            type="primary",
            width="stretch",
        )

    with col2:
        gpu_info = check_gpu_available()
        describe_clicked = st.button(
            "Describe Figures",
            disabled=not gpu_info["available"],
            width="stretch",
            help="Requires CUDA GPU. Uses Qwen3-VL-2B-Instruct." if not gpu_info["available"] else "Generate VLM descriptions for extracted figures.",
        )

    # ──────────────────────────────────────────
    # Convert Document
    # ──────────────────────────────────────────
    if convert_clicked:
        with st.status("Converting document...", expanded=True) as status:
            st.write("Initializing pipeline...")
            st.write(f"Mode: **{config.pipeline_mode.title()}**")

            try:
                st.write("Running document conversion...")
                result = convert_document(file_path, config)
                st.session_state.conversion_result = result

                st.write("Extracting figures...")
                figures_dir = get_figures_dir(st.session_state.session_id)
                figures = extract_figures(result.document, str(figures_dir))
                st.session_state.figures = figures

                # Reset descriptions and enriched markdown
                st.session_state.descriptions = None
                st.session_state.description_result = None
                st.session_state.enriched_md = None

                placeholder_count = count_image_placeholders(result.markdown)

                status.update(
                    label=f"Conversion complete — {result.conversion_time:.1f}s, "
                          f"{len(figures)} figures, {placeholder_count} image placeholders",
                    state="complete",
                )

            except Exception as e:
                status.update(label="Conversion failed", state="error")
                st.error(f"Error during conversion: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="text")

    # ──────────────────────────────────────────
    # Describe Figures with Qwen3-VL
    # ──────────────────────────────────────────
    if describe_clicked:
        if not st.session_state.conversion_result:
            st.warning("⚠️ Please click the **Convert Document** button on the left to process the document first.")
        elif not st.session_state.figures:
            st.info("ℹ️ No figures or images were found in this document to describe.")
        else:
            from pipeline.vlm_describer import describe_figures, load_model

            with st.status("Generating figure descriptions...", expanded=True) as status:
                # Load model (cached in session state, reloaded if model ID changes)
                if (
                    st.session_state.vlm_model is None
                    or getattr(st.session_state, "vlm_model_id", None) != config.downstream_model
                ):
                    st.write(f"Loading {config.downstream_model}...")
                    
                    # Clear previous model to prevent RAM/VRAM leak
                    if st.session_state.vlm_model is not None:
                        del st.session_state.vlm_model
                        del st.session_state.vlm_processor
                        st.session_state.vlm_model = None
                        st.session_state.vlm_processor = None
                        import gc
                        gc.collect()
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except ImportError:
                            pass

                    model, processor = load_model(model_id=config.downstream_model)
                    st.session_state.vlm_model = model
                    st.session_state.vlm_processor = processor
                    st.session_state.vlm_model_id = config.downstream_model
                    st.write("Model loaded.")
                else:
                    st.write(f"Using cached model: {config.downstream_model}")

                st.write(f"Describing {len(st.session_state.figures)} figures...")

                desc_result = describe_figures(
                    st.session_state.figures,
                    model=st.session_state.vlm_model,
                    processor=st.session_state.vlm_processor,
                    model_id=config.downstream_model,
                )

                st.session_state.descriptions = desc_result.descriptions
                st.session_state.description_result = desc_result

                # Auto-enrich markdown
                if st.session_state.conversion_result:
                    figure_paths = [f.image_path for f in st.session_state.figures]
                    enriched = enrich_markdown(
                        st.session_state.conversion_result.markdown,
                        desc_result.descriptions,
                        figure_paths,
                    )
                    st.session_state.enriched_md = enriched

                status.update(
                    label=f"Description complete — {desc_result.inference_time:.1f}s, "
                          f"GPU: {desc_result.gpu_used}",
                    state="complete",
                )

# ──────────────────────────────────────────────
# Step 3: Results Display
# ──────────────────────────────────────────────
if st.session_state.conversion_result:
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    result_tabs = st.tabs([
        "Markdown Output",
        "Extracted Figures",
        "Pipeline Info",
    ])

    # Tab 1: Markdown
    with result_tabs[0]:
        render_markdown_viewer(
            st.session_state.conversion_result.markdown,
            st.session_state.enriched_md,
        )

    # Tab 2: Figures
    with result_tabs[1]:
        if st.session_state.figures is not None:
            render_figure_gallery(
                st.session_state.figures,
                st.session_state.descriptions,
            )
        else:
            st.info("Convert a document to see extracted figures.")

    # Tab 3: Pipeline Info
    with result_tabs[2]:
        render_pipeline_info(
            conversion_result=st.session_state.conversion_result,
            figures_count=len(st.session_state.figures or []),
            description_result=st.session_state.description_result,
        )

else:
    # Welcome state — show pipeline info
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="glass-card animate-in">
                <h3 style="margin-top:0;">How It Works</h3>
                <div class="pipeline-flow">
                    <div class="pipeline-stage">
                        <span class="stage-name">Upload Document</span>
                        <span class="stage-status">PDF, DOCX, PPTX, HTML, Images</span>
                    </div>
                    <div class="pipeline-stage">
                        <span class="stage-name">Configure Pipeline</span>
                        <span class="stage-status">Choose options in the sidebar</span>
                    </div>
                    <div class="pipeline-stage">
                        <span class="stage-name">Convert</span>
                        <span class="stage-status">OCR, tables, formulas, figures</span>
                    </div>
                    <div class="pipeline-stage">
                        <span class="stage-name">Describe Figures</span>
                        <span class="stage-status">Qwen3-VL visual descriptions</span>
                    </div>
                    <div class="pipeline-stage">
                        <span class="stage-name">Enriched Markdown</span>
                        <span class="stage-status">Ready for RAG pipeline</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        gpu_info = check_gpu_available()
        gpu_status = "active" if gpu_info["available"] else "warning"
        gpu_label = "Available" if gpu_info["available"] else "Not Available"

        st.markdown(
            f"""
            <div class="glass-card animate-in">
                <h3 style="margin-top:0;">System Status</h3>
                <div style="display:flex; flex-direction:column; gap:16px; margin-top:1rem;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="status-dot active"></span>
                        <span style="font-size:0.88rem; color:#f0f0f3;">HuggingFace</span>
                        <span style="margin-left:auto; font-size:0.78rem; color:#34d399;
                               font-family:'JetBrains Mono',monospace;">
                            {'Connected' if st.session_state.hf_logged_in else 'Not Connected'}
                        </span>
                    </div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="status-dot {gpu_status}"></span>
                        <span style="font-size:0.88rem; color:#f0f0f3;">GPU ({gpu_info['device']})</span>
                        <span style="margin-left:auto; font-size:0.78rem; color:{'#34d399' if gpu_info['available'] else '#fbbf24'};
                               font-family:'JetBrains Mono',monospace;">
                            {gpu_label}
                        </span>
                    </div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="status-dot active"></span>
                        <span style="font-size:0.88rem; color:#f0f0f3;">Docling</span>
                        <span style="margin-left:auto; font-size:0.78rem; color:#34d399;
                               font-family:'JetBrains Mono',monospace;">
                            Installed
                        </span>
                    </div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="status-dot {'active' if gpu_info['available'] else 'inactive'}"></span>
                        <span style="font-size:0.88rem; color:#f0f0f3;">Qwen3-VL</span>
                        <span style="margin-left:auto; font-size:0.78rem;
                               color:{'#34d399' if gpu_info['available'] else '#5c5c6e'};
                               font-family:'JetBrains Mono',monospace;">
                            {'Ready' if gpu_info['available'] else 'Requires GPU'}
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Supported formats
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card animate-in">
            <h3 style="margin-top:0;">Supported Formats</h3>
            <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:0.8rem;">
                <span style="background:rgba(52,211,153,0.1); color:#34d399; padding:4px 14px;
                       border-radius:999px; font-size:0.78rem; font-family:'JetBrains Mono',monospace;
                       border:1px solid rgba(52,211,153,0.15);">PDF</span>
                <span style="background:rgba(96,165,250,0.1); color:#60a5fa; padding:4px 14px;
                       border-radius:999px; font-size:0.78rem; font-family:'JetBrains Mono',monospace;
                       border:1px solid rgba(96,165,250,0.15);">DOCX</span>
                <span style="background:rgba(251,191,36,0.1); color:#fbbf24; padding:4px 14px;
                       border-radius:999px; font-size:0.78rem; font-family:'JetBrains Mono',monospace;
                       border:1px solid rgba(251,191,36,0.15);">PPTX</span>
                <span style="background:rgba(248,113,113,0.1); color:#f87171; padding:4px 14px;
                       border-radius:999px; font-size:0.78rem; font-family:'JetBrains Mono',monospace;
                       border:1px solid rgba(248,113,113,0.15);">HTML</span>
                <span style="background:rgba(167,139,250,0.1); color:#a78bfa; padding:4px 14px;
                       border-radius:999px; font-size:0.78rem; font-family:'JetBrains Mono',monospace;
                       border:1px solid rgba(167,139,250,0.15);">PNG</span>
                <span style="background:rgba(167,139,250,0.1); color:#a78bfa; padding:4px 14px;
                       border-radius:999px; font-size:0.78rem; font-family:'JetBrains Mono',monospace;
                       border:1px solid rgba(167,139,250,0.15);">JPG</span>
                <span style="background:rgba(167,139,250,0.1); color:#a78bfa; padding:4px 14px;
                       border-radius:999px; font-size:0.78rem; font-family:'JetBrains Mono',monospace;
                       border:1px solid rgba(167,139,250,0.15);">TIFF</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

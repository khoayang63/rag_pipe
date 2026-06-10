"""
Docling RAG Pipeline — Streamlit Application

Main entry point for the document processing pipeline with web UI.
Supports PDF, DOCX, PPTX, HTML, and image files.
Supports both single and multi-file batch conversion.

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
from ui.chunking_viewer import render_chunking_viewer
from ui.vector_store_viewer import render_vector_store_viewer
from pipeline.converter import convert_document, convert_documents
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

# Multi-document results: list of dicts, one per converted document
# Each dict: {"result": ConversionResult, "figures": [...], "descriptions": [...],
#             "description_result": ..., "enriched_md": ...}
if "doc_results" not in st.session_state:
    st.session_state.doc_results = []

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
# Step 1: File Upload (multi-file)
# ──────────────────────────────────────────────
file_paths = render_upload()

# ──────────────────────────────────────────────
# Step 2: Convert Button
# ──────────────────────────────────────────────
if file_paths:
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    convert_clicked = st.button(
        f"Convert {'Documents' if len(file_paths) > 1 else 'Document'} ({len(file_paths)} file{'s' if len(file_paths) > 1 else ''})",
        type="primary",
        use_container_width=False,
    )

    # ──────────────────────────────────────────
    # Convert Documents
    # ──────────────────────────────────────────
    if convert_clicked:
        with st.status(f"Converting {len(file_paths)} document(s)...", expanded=True) as status:
            st.write("Initializing pipeline...")
            st.write(f"Mode: **{config.pipeline_mode.title()}**")

            try:
                if len(file_paths) == 1:
                    # Single file — use convert_document for exact backward compat
                    st.write(f"Converting: **{os.path.basename(file_paths[0])}**")
                    result = convert_document(file_paths[0], config)
                    conversion_results = [result]
                else:
                    # Multiple files — use convert_all for batch efficiency
                    st.write(f"Batch converting {len(file_paths)} files...")
                    conversion_results = convert_documents(
                        file_paths,
                        config,
                        progress_callback=lambda i, t, name: st.write(f"[{i}/{t}] Converting: **{name}**"),
                    )

                # Build per-document results with figure extraction
                doc_results = []
                figures_dir = get_figures_dir(st.session_state.session_id)

                for idx, conv_result in enumerate(conversion_results):
                    st.write(f"Extracting figures from: **{conv_result.source_filename}**")

                    # Each doc gets its own figures subfolder
                    doc_figures_dir = figures_dir / f"doc_{idx}"
                    doc_figures_dir.mkdir(parents=True, exist_ok=True)

                    figures = extract_figures(conv_result.document, str(doc_figures_dir))
                    placeholder_count = count_image_placeholders(conv_result.markdown)

                    doc_results.append({
                        "result": conv_result,
                        "figures": figures,
                        "descriptions": None,
                        "description_result": None,
                        "enriched_md": None,
                        "placeholder_count": placeholder_count,
                    })

                st.session_state.doc_results = doc_results

                total_figures = sum(len(d["figures"]) for d in doc_results)
                total_time = sum(r.conversion_time for r in conversion_results)

                status.update(
                    label=f"Conversion complete — {total_time:.1f}s, "
                          f"{len(conversion_results)} doc(s), {total_figures} figures",
                    state="complete",
                )

            except Exception as e:
                status.update(label="Conversion failed", state="error")
                st.error(f"Error during conversion: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="text")

# ──────────────────────────────────────────────
# Step 3: Results Display (per-document)
# ──────────────────────────────────────────────
if st.session_state.doc_results:
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Summary metrics
    total_docs = len(st.session_state.doc_results)
    total_figures = sum(len(d["figures"]) for d in st.session_state.doc_results)
    total_time = sum(d["result"].conversion_time for d in st.session_state.doc_results)

    st.markdown(
        f"""
        <div class="glass-card animate-in" style="padding:1rem 1.5rem; margin-bottom:1rem;">
            <div style="display:flex; gap:2rem; align-items:center;">
                <div style="text-align:center;">
                    <div style="font-size:1.5rem; font-weight:700; color:#34d399;
                         font-family:'JetBrains Mono',monospace;">{total_docs}</div>
                    <div style="font-size:0.72rem; color:#5c5c6e; text-transform:uppercase;
                         letter-spacing:0.05em;">Documents</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.5rem; font-weight:700; color:#60a5fa;
                         font-family:'JetBrains Mono',monospace;">{total_figures}</div>
                    <div style="font-size:0.72rem; color:#5c5c6e; text-transform:uppercase;
                         letter-spacing:0.05em;">Figures</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.5rem; font-weight:700; color:#fbbf24;
                         font-family:'JetBrains Mono',monospace;">{total_time:.1f}s</div>
                    <div style="font-size:0.72rem; color:#5c5c6e; text-transform:uppercase;
                         letter-spacing:0.05em;">Total Time</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Per-document expanders
    for doc_idx, doc_data in enumerate(st.session_state.doc_results):
        result = doc_data["result"]
        figures = doc_data["figures"]
        filename = result.source_filename or f"Document {doc_idx + 1}"

        fig_count = len(figures)
        time_str = f"{result.conversion_time:.1f}s"

        with st.expander(
            f"📄 {filename}  —  {fig_count} figures  ·  {time_str}",
            expanded=(total_docs == 1),
        ):
            # ── Per-document Describe Figures button ──
            gpu_info = check_gpu_available()
            if figures:
                describe_col1, describe_col2 = st.columns([1, 4])
                with describe_col1:
                    describe_clicked = st.button(
                        "Describe Figures",
                        key=f"describe_{doc_idx}",
                        disabled=not gpu_info["available"],
                        help=(
                            "Requires CUDA GPU. Uses Qwen3-VL-2B-Instruct."
                            if not gpu_info["available"]
                            else f"Generate VLM descriptions for {fig_count} figures in this document."
                        ),
                    )

                if describe_clicked:
                    from pipeline.vlm_describer import describe_figures, load_model

                    with st.status(f"Describing figures for {filename}...", expanded=True) as desc_status:
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

                        st.write(f"Describing {len(figures)} figures...")

                        desc_result = describe_figures(
                            figures,
                            model=st.session_state.vlm_model,
                            processor=st.session_state.vlm_processor,
                            model_id=config.downstream_model,
                        )

                        # Update this document's results
                        doc_data["descriptions"] = desc_result.descriptions
                        doc_data["description_result"] = desc_result

                        # Auto-enrich markdown
                        figure_paths = [f.image_path for f in figures]
                        enriched = enrich_markdown(
                            result.markdown,
                            desc_result.descriptions,
                            figure_paths,
                        )
                        doc_data["enriched_md"] = enriched

                        desc_status.update(
                            label=f"Description complete — {desc_result.inference_time:.1f}s, "
                                  f"GPU: {desc_result.gpu_used}",
                            state="complete",
                        )

            # ── Per-document result tabs ──
            result_tabs = st.tabs([
                "Markdown Output",
                "Extracted Figures",
                "Chunking",
                "Pipeline Info",
                "Vector DB",
            ])

            # Tab 1: Markdown
            with result_tabs[0]:
                render_markdown_viewer(
                    result.markdown,
                    doc_data.get("enriched_md"),
                    doc_id=str(doc_idx),
                )

            # Tab 2: Figures
            with result_tabs[1]:
                if figures:
                    render_figure_gallery(
                        figures,
                        doc_data.get("descriptions"),
                    )
                else:
                    st.info("No figures found in this document.")

            # Tab 3: Chunking
            with result_tabs[2]:
                render_chunking_viewer(
                    document=result.document,
                    doc_id=str(doc_idx),
                )

            # Tab 4: Pipeline Info
            with result_tabs[3]:
                render_pipeline_info(
                    conversion_result=result,
                    figures_count=len(figures),
                    description_result=doc_data.get("description_result"),
                )

            # Tab 5: Vector DB
            with result_tabs[4]:
                render_vector_store_viewer(
                    doc_name=filename,
                    doc_id=str(doc_idx),
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
                        <span class="stage-name">Upload Documents</span>
                        <span class="stage-status">PDF, DOCX, PPTX, HTML, Images (multi-file)</span>
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

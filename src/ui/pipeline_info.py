"""
Pipeline information component.

Displays pipeline architecture, models used at each stage,
processing times, and GPU status.
"""

import streamlit as st
from config import check_gpu_available


def clean_html(html_str: str) -> str:
    """Remove leading whitespace from each line in a multiline HTML string to prevent Markdown code block parsing."""
    return "\n".join(line.lstrip() for line in html_str.splitlines())


def render_pipeline_info(
    conversion_result=None,
    figures_count: int = 0,
    description_result=None,
):
    """
    Render the pipeline information panel.

    Shows:
    - Pipeline architecture flow diagram
    - Models used at each stage
    - Processing times
    - GPU memory info
    """
    gpu_info = check_gpu_available()

    # GPU Status Card
    st.markdown(
        clean_html(f"""
        <div class="glass-card animate-in" style="margin-bottom:1.5rem;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
                <span class="status-dot {'active' if gpu_info['available'] else 'warning'}"></span>
                <h3 style="margin:0; font-size:1rem;">GPU Status</h3>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                <div>
                    <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;
                       color:#5c5c6e; margin:0 0 2px 0;">Device</p>
                    <p style="font-size:0.88rem; color:#f0f0f3; margin:0;
                       font-family:'JetBrains Mono',monospace;">
                        {gpu_info['device']}
                    </p>
                </div>
                <div>
                    <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;
                       color:#5c5c6e; margin:0 0 2px 0;">VRAM</p>
                    <p style="font-size:0.88rem; color:#34d399; margin:0;
                       font-family:'JetBrains Mono',monospace;">
                        {gpu_info['vram_gb']} GB
                    </p>
                </div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # Pipeline Architecture Flow
    flow_html = ""
    if conversion_result and conversion_result.pipeline_mode == "vlm":
        flow_html = _get_vlm_flow_html(conversion_result)
    else:
        flow_html = _get_standard_flow_html(conversion_result)

    st.markdown(
        clean_html(f"""
        <div class="glass-card animate-in" style="margin-bottom:1.5rem;">
            <h3 style="margin:0 0 1rem 0; font-size:1rem;">Pipeline Architecture</h3>
            {flow_html}
        </div>
        """),
        unsafe_allow_html=True,
    )

    # Processing Metrics
    if conversion_result:
        _render_metrics(conversion_result, figures_count, description_result)

    # Model Details
    _render_model_details(conversion_result, description_result)


def _get_standard_flow_html(conversion_result=None) -> str:
    """Return the Standard pipeline flow diagram HTML."""
    desc_model_id = "Qwen/Qwen3-VL-2B-Instruct"
    ocr_engine = "easyocr"
    actual_ocr = None
    layout_model = "heron"

    if conversion_result:
        desc_model_id = conversion_result.pipeline_config.get("downstream_model", desc_model_id)
        ocr_engine = conversion_result.pipeline_config.get("ocr_engine", ocr_engine)
        actual_ocr = conversion_result.pipeline_config.get("actual_ocr", None)
        layout_model = conversion_result.pipeline_config.get("layout_model", layout_model)
    
    desc_display = desc_model_id.split("/")[-1].replace("-Instruct", "")

    ocr_display_map = {
        "easyocr": "EasyOCR",
        "rapidocr": "RapidOCR",
        "tesseract": "Tesseract OCR",
        "tesseract_cli": "Tesseract CLI OCR",
        "macocr": "Mac OCR",
        "kserve": "Kserve V2 OCR",
        "auto": "Auto OCR",
    }
    
    layout_display_map = {
        "layout_v2": "Layout V2",
        "heron": "Layout Heron",
        "heron_101": "Layout Heron-101",
        "egret_medium": "Layout Egret Medium",
        "egret_large": "Layout Egret Large",
        "egret_xlarge": "Layout Egret XLarge",
    }
    
    ocr_display = ocr_display_map.get(ocr_engine.lower(), "EasyOCR")
    if ocr_engine.lower() == "auto" and actual_ocr and actual_ocr != "unknown":
        actual_display = ocr_display_map.get(actual_ocr.lower(), actual_ocr.title())
        ocr_display = f"Auto (Detected: {actual_display})"
        
    layout_display = layout_display_map.get(layout_model.lower(), "Layout Heron")

    stages = [
        ("Document Input", "DocumentConverter", "Detects format, routes to pipeline"),
        ("Layout Analysis", layout_display, "Page segmentation and region classification"),
        ("OCR", ocr_display, "Text recognition from scanned content"),
        ("Table Extraction", "TableFormer", "Table structure and cell content extraction"),
        ("Formula Extraction", "Built-in", "Mathematical formula detection and LaTeX"),
        ("Code Extraction", "Built-in", "Code block detection and language tagging"),
        ("Figure Extraction", "Built-in", "Picture cropping and provenance tracking"),
        ("Markdown Export", "Docling Core", "Structured document to markdown conversion"),
        ("Figure Description", desc_display, "VLM-based visual description generation"),
        ("Markdown Enrichment", "Post-processing", "Insert figure images and descriptions"),
    ]

    flow_html = '<div class="pipeline-flow">'
    for name, model, desc in stages:
        flow_html += f'<div class="pipeline-stage"><span class="stage-name">{name}</span><span class="stage-model">{model}</span><span class="stage-status">{desc}</span></div>'
    flow_html += "</div>"
    return flow_html


def _get_vlm_flow_html(conversion_result=None) -> str:
    """Return the VLM pipeline flow diagram HTML."""
    preset = "Granite Vision"
    if conversion_result:
        cfg = conversion_result.pipeline_config
        preset = cfg.get("vlm_preset", "Granite Vision").replace("_", " ").title()

    stages = [
        ("Document Input", "DocumentConverter", "Format detection and routing"),
        ("Page Rendering", "VlmPipeline", "Page-level image rendering"),
        ("VLM Inference", preset, "End-to-end vision-language conversion"),
        ("Markdown Export", "Docling Core", "Structured output to markdown"),
    ]

    flow_html = '<div class="pipeline-flow">'
    for name, model, desc in stages:
        flow_html += f'<div class="pipeline-stage"><span class="stage-name">{name}</span><span class="stage-model">{model}</span><span class="stage-status">{desc}</span></div>'
    flow_html += "</div>"
    return flow_html


def _render_metrics(conversion_result, figures_count, description_result):
    """Render processing time metrics using custom HTML cards."""
    conv_time = f"{conversion_result.conversion_time:.1f}s"
    fig_count = f"{figures_count}"
    
    if description_result:
        vlm_time = f"{description_result.inference_time:.1f}s"
    else:
        vlm_time = "—"
        
    input_fmt = conversion_result.input_format.upper()
    pipeline_type = conversion_result.pipeline_mode.title()
    md_size = f"{len(conversion_result.markdown) / 1024:.1f} KB"

    metrics_html = f"""
    <div class="glass-card animate-in" style="margin-bottom:1.5rem;">
        <h3 style="margin:0 0 1.2rem 0; font-size:1rem;">Processing Metrics</h3>
        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px;">
            <div style="background: rgba(255,255,255,0.02); padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04);">
                <p style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; margin:0 0 4px 0;">Conversion Time</p>
                <p style="font-size:1.4rem; color:var(--accent); font-weight:600; margin:0; font-family:'JetBrains Mono',monospace;">{conv_time}</p>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04);">
                <p style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; margin:0 0 4px 0;">Figures Extracted</p>
                <p style="font-size:1.4rem; color:var(--accent); font-weight:600; margin:0; font-family:'JetBrains Mono',monospace;">{fig_count}</p>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04);">
                <p style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; margin:0 0 4px 0;">VLM Inference</p>
                <p style="font-size:1.4rem; color:var(--accent); font-weight:600; margin:0; font-family:'JetBrains Mono',monospace;">{vlm_time}</p>
            </div>
        </div>
        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
            <div style="background: rgba(255,255,255,0.02); padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04);">
                <p style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; margin:0 0 4px 0;">Input Format</p>
                <p style="font-size:1.4rem; color:var(--accent); font-weight:600; margin:0; font-family:'JetBrains Mono',monospace;">{input_fmt}</p>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04);">
                <p style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; margin:0 0 4px 0;">Pipeline</p>
                <p style="font-size:1.4rem; color:var(--accent); font-weight:600; margin:0; font-family:'JetBrains Mono',monospace;">{pipeline_type}</p>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04);">
                <p style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; margin:0 0 4px 0;">Markdown Size</p>
                <p style="font-size:1.4rem; color:var(--accent); font-weight:600; margin:0; font-family:'JetBrains Mono',monospace;">{md_size}</p>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(metrics_html), unsafe_allow_html=True)


def _render_model_details(conversion_result, description_result):
    """Render detailed model information inside a single HTML block."""
    desc_model_id = "Qwen/Qwen3-VL-2B-Instruct"
    ocr_engine = "easyocr"
    actual_ocr = None
    layout_model = "heron"

    if conversion_result:
        desc_model_id = conversion_result.pipeline_config.get("downstream_model", desc_model_id)
        ocr_engine = conversion_result.pipeline_config.get("ocr_engine", ocr_engine)
        actual_ocr = conversion_result.pipeline_config.get("actual_ocr", None)
        layout_model = conversion_result.pipeline_config.get("layout_model", layout_model)
    
    desc_display = desc_model_id.split("/")[-1]
    
    desc_type = "Vision-Language Model"
    if "3B" in desc_display:
        desc_type = "Vision-Language Model (3B params)"
    elif "2B" in desc_display:
        desc_type = "Vision-Language Model (2B params)"
    elif "256M" in desc_display:
        desc_type = "Vision-Language Model (256M params)"

    ocr_display_map = {
        "easyocr": "EasyOCR",
        "rapidocr": "RapidOCR",
        "tesseract": "Tesseract OCR",
        "tesseract_cli": "Tesseract CLI OCR",
        "macocr": "Mac OCR",
        "kserve": "Kserve V2 OCR",
        "auto": "Auto OCR",
    }
    ocr_type_map = {
        "easyocr": "CNN + LSTM",
        "rapidocr": "CNN + GRU",
        "tesseract": "LSTM",
        "tesseract_cli": "LSTM",
        "macocr": "Apple Native OCR",
        "kserve": "API Endpoint",
        "auto": "Auto Select",
    }

    layout_display_map = {
        "layout_v2": "Layout V2",
        "heron": "Layout Heron",
        "heron_101": "Layout Heron-101",
        "egret_medium": "Layout Egret Medium",
        "egret_large": "Layout Egret Large",
        "egret_xlarge": "Layout Egret XLarge",
    }
    layout_type_map = {
        "layout_v2": "Vision Transformer",
        "heron": "Vision Transformer (Heron)",
        "heron_101": "Vision Transformer (Heron-101)",
        "egret_medium": "Vision Transformer (Egret Medium)",
        "egret_large": "Vision Transformer (Egret Large)",
        "egret_xlarge": "Vision Transformer (Egret XLarge)",
    }

    ocr_display = ocr_display_map.get(ocr_engine.lower(), "EasyOCR")
    ocr_type = ocr_type_map.get(ocr_engine.lower(), "CNN + LSTM")
    if ocr_engine.lower() == "auto" and actual_ocr and actual_ocr != "unknown":
        actual_display = ocr_display_map.get(actual_ocr.lower(), actual_ocr.title())
        ocr_display = f"Auto (Detected: {actual_display})"
        ocr_type = ocr_type_map.get(actual_ocr.lower(), ocr_type)

    layout_display = layout_display_map.get(layout_model.lower(), "Layout Heron")
    layout_type = layout_type_map.get(layout_model.lower(), "Vision Transformer")

    # Check if selected OCR engine is installed
    ocr_installed = True
    check_engine = actual_ocr if (ocr_engine.lower() == "auto" and actual_ocr and actual_ocr != "unknown") else ocr_engine
    if check_engine and check_engine.lower() == "easyocr":
        try:
            import easyocr
        except ImportError:
            ocr_installed = False
    elif check_engine and check_engine.lower() == "rapidocr":
        try:
            import rapidocr
        except ImportError:
            ocr_installed = False
    elif check_engine and check_engine.lower() in ["tesseract", "tesseract_cli"]:
        try:
            import pytesseract
        except ImportError:
            ocr_installed = False

    ocr_badge = ocr_display
    if not ocr_installed:
        ocr_badge = f"{ocr_display} <span style='color:#ef4444; font-size:0.7rem; font-weight:600; margin-left:6px; background:rgba(239,68,68,0.1); padding:2px 6px; border-radius:4px; border:1px solid rgba(239,68,68,0.2);'>⚠️ Not Installed</span>"

    models = [
        {
            "name": "Layout Analysis",
            "model": layout_display,
            "purpose": "Page segmentation, region classification",
            "type": layout_type,
        },
        {
            "name": "OCR Engine",
            "model": ocr_badge,
            "purpose": "Text recognition from images",
            "type": ocr_type,
        },
        {
            "name": "Table Structure",
            "model": "TableFormer",
            "purpose": "Table cell detection and structure",
            "type": "Transformer",
        },
        {
            "name": "Figure Description",
            "model": desc_display,
            "purpose": "Visual description generation",
            "type": desc_type,
        },
    ]

    if conversion_result and conversion_result.pipeline_mode == "vlm":
        cfg = conversion_result.pipeline_config
        vlm_name = cfg.get("vlm_preset", "Unknown").replace("_", " ").title()
        models = [
            {
                "name": "VLM Model",
                "model": vlm_name,
                "purpose": "End-to-end document understanding",
                "type": "Vision-Language Model",
            },
        ]

    rows_html = ""
    for m in models:
        rows_html += f"""
        <div style="
            display:flex; align-items:flex-start; gap:12px;
            padding:12px 0;
            border-bottom:1px solid rgba(255,255,255,0.04);
        ">
            <div style="min-width:140px;">
                <span style="font-size:0.82rem; color:#f0f0f3; font-weight:500;">
                    {m['name']}
                </span>
            </div>
            <div style="flex:1;">
                <span style="
                    font-family:'JetBrains Mono',monospace;
                    font-size:0.78rem; color:#34d399;
                    background:rgba(52,211,153,0.1);
                    padding:2px 8px; border-radius:999px;
                ">{m['model']}</span>
            </div>
            <div style="flex:1;">
                <span style="font-size:0.78rem; color:#5c5c6e;">{m['purpose']}</span>
            </div>
        </div>
        """

    details_html = f"""
    <div class="glass-card animate-in">
        <h3 style="margin:0 0 1rem 0; font-size:1rem;">Model Details</h3>
        {rows_html}
    </div>
    """
    st.markdown(clean_html(details_html), unsafe_allow_html=True)

"""
File upload component.

Provides a drag-and-drop file uploader supporting multiple
document formats: PDF, DOCX, PPTX, HTML, and various image types.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path

from config import SUPPORTED_FORMATS, ACCEPTED_EXTENSIONS


def render_upload() -> str | None:
    """
    Render the file upload component.

    Returns:
        Path to the uploaded file (saved to temp), or None if no file.
    """
    st.markdown(
        """
        <div class="glass-card animate-in" style="text-align:center; padding:2.5rem;">
            <div style="font-size:2.5rem; margin-bottom:0.8rem; opacity:0.6;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
                     stroke="#34d399" stroke-width="1.5" stroke-linecap="round"
                     stroke-linejoin="round" style="display:inline-block;">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
            </div>
            <h3 style="margin:0 0 0.3rem 0; font-size:1.1rem; color:#f0f0f3;">
                Upload Document
            </h3>
            <p style="margin:0; font-size:0.82rem; color:#5c5c6e;">
                PDF, DOCX, PPTX, HTML, PNG, JPG, TIFF — Drag and drop or click to browse
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=[ext.lstrip(".") for ext in ACCEPTED_EXTENSIONS],
        label_visibility="collapsed",
        help="Supports: " + ", ".join(SUPPORTED_FORMATS.values()),
    )

    if uploaded_file is not None:
        # Display file info
        file_ext = Path(uploaded_file.name).suffix.lower()
        file_type = SUPPORTED_FORMATS.get(file_ext, "Unknown")
        file_size_kb = uploaded_file.size / 1024
        file_size_display = (
            f"{file_size_kb:.1f} KB"
            if file_size_kb < 1024
            else f"{file_size_kb / 1024:.1f} MB"
        )

        st.markdown(
            f"""
            <div class="glass-card animate-in" style="margin-top:1rem;">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="
                        width:40px; height:40px;
                        background: rgba(52, 211, 153, 0.1);
                        border-radius: 10px;
                        display:flex; align-items:center; justify-content:center;
                        color: #34d399; font-weight:600; font-size:0.7rem;
                        text-transform: uppercase;
                    ">{file_ext.lstrip('.') if len(file_ext) <= 5 else 'DOC'}</div>
                    <div>
                        <p style="margin:0; font-weight:600; color:#f0f0f3; font-size:0.92rem;">
                            {uploaded_file.name}
                        </p>
                        <p style="margin:2px 0 0 0; font-size:0.78rem; color:#5c5c6e;">
                            {file_type} &middot; {file_size_display}
                        </p>
                    </div>
                    <div style="margin-left:auto;">
                        <span class="status-dot active"></span>
                        <span style="font-size:0.78rem; color:#34d399;">Ready</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Save to temp file
        tmp_dir = tempfile.mkdtemp(prefix="docling_upload_")
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return tmp_path

    return None

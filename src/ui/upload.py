"""
File upload component.

Provides a drag-and-drop file uploader supporting multiple
document formats: PDF, DOCX, PPTX, HTML, and various image types.
Supports both single and multi-file upload.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path

from config import SUPPORTED_FORMATS, ACCEPTED_EXTENSIONS


def render_upload() -> list[str]:
    """
    Render the file upload component with multi-file support and deletion capability.

    Returns:
        List of paths to uploaded files (saved to temp), or empty list if none.
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
                Upload Documents
            </h3>
            <p style="margin:0; font-size:0.82rem; color:#5c5c6e;">
                PDF, DOCX, PPTX, XLSX, CSV, HTML, PNG, JPG, TIFF — Drag and drop or click to browse (multiple files supported)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=None,  # Allow all files at browser level to bypass Streamlit MIME validation bugs
        label_visibility="collapsed",
        help="Supports: " + ", ".join(SUPPORTED_FORMATS.values()),
        accept_multiple_files=True,
    )

    # Initialize state variables
    if "deleted_files" not in st.session_state:
        st.session_state.deleted_files = set()
    if "upload_cache" not in st.session_state:
        st.session_state.upload_cache = {}

    if uploaded_files:
        # Filter files by accepted extensions to bypass any browser MIME bugs
        valid_uploaded_files = []
        invalid_uploaded_files = []
        for f in uploaded_files:
            file_ext = Path(f.name).suffix.lower()
            if file_ext in ACCEPTED_EXTENSIONS:
                valid_uploaded_files.append(f)
            else:
                invalid_uploaded_files.append(f)

        # Show warning/error for unsupported formats
        if invalid_uploaded_files:
            unsupported_names = ", ".join([f.name for f in invalid_uploaded_files])
            st.error(f"Tệp không được hỗ trợ: {unsupported_names}. Định dạng hỗ trợ: " + ", ".join(SUPPORTED_FORMATS.values()))

        if not valid_uploaded_files:
            return []

        # Clean up deleted_files for files no longer present in the uploader widget
        uploaded_names = {f.name for f in valid_uploaded_files}
        st.session_state.deleted_files = {
            name for name in st.session_state.deleted_files if name in uploaded_names
        }

        # Filter active files from valid uploaded files
        active_files = [f for f in valid_uploaded_files if f.name not in st.session_state.deleted_files]

        if not active_files:
            return []

        # Summary card for active files
        total_size = sum(f.size for f in active_files)
        total_size_display = (
            f"{total_size / 1024:.1f} KB"
            if total_size < 1024 * 1024
            else f"{total_size / (1024 * 1024):.1f} MB"
        )

        if len(active_files) > 1:
            st.markdown(
                f"""
                <div class="glass-card animate-in" style="margin-top:1rem; padding:0.8rem 1.2rem;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                        <span style="font-size:0.88rem; font-weight:600; color:#f0f0f3;">
                            {len(active_files)} files selected
                        </span>
                        <span style="margin-left:auto; font-size:0.78rem; color:#5c5c6e;
                               font-family:'JetBrains Mono',monospace;">
                            {total_size_display} total
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Individual file cards
        tmp_paths = []
        for idx, uploaded_file in enumerate(active_files):
            file_ext = Path(uploaded_file.name).suffix.lower()
            file_type = SUPPORTED_FORMATS.get(file_ext, "Unknown")
            file_size_kb = uploaded_file.size / 1024
            file_size_display = (
                f"{file_size_kb:.1f} KB"
                if file_size_kb < 1024
                else f"{file_size_kb / 1024:.1f} MB"
            )

            # Render file card using raw HTML (avoids st.container/st.columns issues)
            st.markdown(
                f"""<div class="file-card-v2" style="display: flex; align-items: center; gap: 16px; background: rgba(26, 26, 34, 0.7); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px 16px; margin-top: 8px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.04); transition: all 0.2s ease-in-out; width: 100%; min-width: 0; box-sizing: border-box;"><div style="flex-shrink: 0; width: 38px; height: 38px; background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.2); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #34d399; font-weight: 700; font-size: 0.7rem; letter-spacing: 0.05em; text-transform: uppercase;">{file_ext.lstrip('.') if len(file_ext) <= 5 else 'DOC'}</div><div style="flex-grow: 1; min-width: 0; display: flex; flex-direction: column; justify-content: center;"><div style="margin: 0; font-weight: 600; color: #f0f0f3; font-size: 0.9rem; line-height: 1.4; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{uploaded_file.name}</div><div style="margin: 2px 0 0 0; font-size: 0.78rem; color: #8a8a9e; line-height: 1.3; display: flex; align-items: center; gap: 6px;"><span>{file_type}</span><span style="opacity: 0.4;">&bull;</span><span>{file_size_display}</span><span style="opacity: 0.4;">&bull;</span><span style="color: #34d399; font-weight: 600; background: rgba(52, 211, 153, 0.1); padding: 0px 6px; border-radius: 4px; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.03em;">Ready</span></div></div></div>""",
                unsafe_allow_html=True,
            )

            # Retrieve or create temp path
            cache_key = (uploaded_file.name, uploaded_file.size)
            if cache_key in st.session_state.upload_cache and os.path.exists(st.session_state.upload_cache[cache_key]):
                tmp_path = st.session_state.upload_cache[cache_key]
            else:
                tmp_dir = tempfile.mkdtemp(prefix="docling_upload_")
                tmp_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.upload_cache[cache_key] = tmp_path
                
            tmp_paths.append(tmp_path)

        return tmp_paths

    return []

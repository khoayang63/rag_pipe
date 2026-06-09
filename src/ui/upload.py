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
                PDF, DOCX, PPTX, HTML, PNG, JPG, TIFF — Drag and drop or click to browse (multiple files supported)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[ext.lstrip(".") for ext in ACCEPTED_EXTENSIONS],
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
        # Clean up deleted_files for files no longer present in the uploader widget
        uploaded_names = {f.name for f in uploaded_files}
        st.session_state.deleted_files = {
            name for name in st.session_state.deleted_files if name in uploaded_names
        }

        # Filter active files
        active_files = [f for f in uploaded_files if f.name not in st.session_state.deleted_files]

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
                        <span style="font-size:1.2rem;">📑</span>
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

            # Container for file card
            with st.container(key=f"file_card_{idx}"):
                col_icon, col_info, col_delete = st.columns([0.8, 8.2, 1.0])
                
                with col_icon:
                    st.markdown(
                        f"""
                        <div style="
                            width:36px; height:36px;
                            background: rgba(52, 211, 153, 0.1);
                            border-radius: 8px;
                            display:flex; align-items:center; justify-content:center;
                            color: #34d399; font-weight:600; font-size:0.65rem;
                            text-transform: uppercase;
                        ">{file_ext.lstrip('.') if len(file_ext) <= 5 else 'DOC'}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                with col_info:
                    st.markdown(
                        f"""
                        <div style="display:flex; flex-direction:column; justify-content:center; height:36px;">
                            <p style="margin:0; font-weight:600; color:#f0f0f3; font-size:0.88rem; line-height:1.2;
                                      overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                                {uploaded_file.name}
                            </p>
                            <p style="margin:2px 0 0 0; font-size:0.75rem; color:#5c5c6e; line-height:1.2;">
                                {file_type} &middot; {file_size_display} &middot; <span style="color:#34d399; font-weight:500;">Ready</span>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                with col_delete:
                    # Render delete button at the rightmost
                    if st.button("🗑️", key=f"delete_file_{idx}", help="Remove file"):
                        # Mark as deleted
                        st.session_state.deleted_files.add(uploaded_file.name)
                        
                        # Clean up temp file
                        cache_key = (uploaded_file.name, uploaded_file.size)
                        if cache_key in st.session_state.upload_cache:
                            old_path = st.session_state.upload_cache[cache_key]
                            try:
                                if os.path.exists(old_path):
                                    os.remove(old_path)
                                    parent_dir = os.path.dirname(old_path)
                                    if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                                        os.rmdir(parent_dir)
                            except Exception:
                                pass
                            del st.session_state.upload_cache[cache_key]
                        
                        # Remove matching results from session state doc_results
                        if "doc_results" in st.session_state and st.session_state.doc_results:
                            st.session_state.doc_results = [
                                doc for doc in st.session_state.doc_results
                                if doc["result"].source_filename != uploaded_file.name
                            ]
                        
                        st.rerun()

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

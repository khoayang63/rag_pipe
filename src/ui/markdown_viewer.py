"""
Markdown viewer component.

Provides a tabbed view for rendered markdown and raw source,
with download functionality.
"""

import streamlit as st


def render_markdown_viewer(md: str, enriched_md: str | None = None):
    """
    Render the markdown viewer with Rendered/Raw/Enriched tabs.

    Args:
        md: Raw markdown from Docling conversion
        enriched_md: Optional enriched markdown (after VLM descriptions)
    """
    if enriched_md:
        tabs = st.tabs(["Rendered", "Enriched", "Raw Source"])
    else:
        tabs = st.tabs(["Rendered", "Raw Source"])

    tab_idx = 0

    # Rendered tab
    with tabs[tab_idx]:
        st.markdown(
            f'<div class="markdown-viewer">{_md_to_styled(md)}</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            st.download_button(
                label="Download .md",
                data=md,
                file_name="converted_document.md",
                mime="text/markdown",
            )
        with col2:
            _render_stats(md)
    tab_idx += 1

    # Enriched tab (if available)
    if enriched_md:
        with tabs[tab_idx]:
            st.markdown(
                f'<div class="markdown-viewer">{_md_to_styled(enriched_md)}</div>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns([1, 5])
            with col1:
                st.download_button(
                    label="Download enriched .md",
                    data=enriched_md,
                    file_name="enriched_document.md",
                    mime="text/markdown",
                )
            with col2:
                _render_stats(enriched_md)
        tab_idx += 1

    # Raw source tab
    with tabs[tab_idx]:
        display_md = enriched_md if enriched_md else md
        st.code(display_md, language="markdown", line_numbers=True)
        st.download_button(
            label="Download raw .md",
            data=display_md,
            file_name="raw_document.md",
            mime="text/markdown",
            key="download_raw",
        )


def _md_to_styled(md: str) -> str:
    """Convert markdown to basic HTML for display. Streamlit handles most rendering."""
    # We use st.markdown which handles markdown rendering natively.
    # This wrapper just returns the markdown for the styled div.
    # For the styled container we pass raw markdown and let Streamlit handle it.
    return md


def _render_stats(md: str):
    """Render markdown statistics inline."""
    lines = md.count("\n") + 1
    words = len(md.split())
    chars = len(md)
    tables = md.count("|---") + md.count("| ---")
    images = md.count("<!-- image -->") + md.count("![")

    st.markdown(
        f"""
        <div style="display:flex; gap:1.5rem; align-items:center; padding:0.5rem 0;">
            <span style="font-size:0.75rem; color:#5c5c6e;">
                <span style="color:#9898a6; font-family:'JetBrains Mono',monospace;">{lines:,}</span> lines
            </span>
            <span style="font-size:0.75rem; color:#5c5c6e;">
                <span style="color:#9898a6; font-family:'JetBrains Mono',monospace;">{words:,}</span> words
            </span>
            <span style="font-size:0.75rem; color:#5c5c6e;">
                <span style="color:#9898a6; font-family:'JetBrains Mono',monospace;">{chars:,}</span> chars
            </span>
            <span style="font-size:0.75rem; color:#5c5c6e;">
                <span style="color:#9898a6; font-family:'JetBrains Mono',monospace;">{tables}</span> tables
            </span>
            <span style="font-size:0.75rem; color:#5c5c6e;">
                <span style="color:#9898a6; font-family:'JetBrains Mono',monospace;">{images}</span> images
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_markdown_viewer_simple(md: str):
    """
    Simplified markdown viewer using Streamlit's native markdown rendering.
    Used as the primary rendered view.
    """
    st.markdown(md)

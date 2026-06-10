"""
Figure gallery component.

Displays extracted figures in a grid layout with metadata:
caption, page number, bounding box, and VLM descriptions.
"""

import streamlit as st
from PIL import Image
from pathlib import Path


def render_figure_gallery(
    figures: list,
    descriptions: list[str] | None = None,
):
    """
    Render the figure gallery grid.

    Args:
        figures: List of FigureData objects
        descriptions: Optional list of VLM-generated descriptions
    """
    if not figures:
        # Empty state
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:3rem;">
                <div style="font-size:2rem; margin-bottom:0.5rem; opacity:0.3;">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
                         stroke="#5c5c6e" stroke-width="1.5" stroke-linecap="round"
                         stroke-linejoin="round" style="display:inline-block;">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                        <circle cx="8.5" cy="8.5" r="1.5"/>
                        <polyline points="21 15 16 10 5 21"/>
                    </svg>
                </div>
                <h3 style="margin:0; font-size:1rem; color:#5c5c6e;">No Figures Extracted</h3>
                <p style="margin:0.3rem 0 0; font-size:0.82rem; color:#3d3d4a;">
                    Convert a document with "Generate Picture Images" enabled to extract figures.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Header with count
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:1rem;">
            <span class="status-dot active"></span>
            <span style="font-size:0.85rem; color:#9898a6;">
                <span style="color:#f0f0f3; font-weight:600;">{len(figures)}</span> figures extracted
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Grid layout — 3 columns on desktop
    cols_per_row = 3
    for row_start in range(0, len(figures), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            fig_idx = row_start + col_idx
            if fig_idx >= len(figures):
                break

            fig = figures[fig_idx]
            desc = descriptions[fig_idx] if descriptions and fig_idx < len(descriptions) else None

            with cols[col_idx]:
                _render_figure_card(fig, desc)


def _render_figure_card(fig, description: str | None = None):
    """Render a single figure card with metadata."""
    # Load and display image
    try:
        img = Image.open(fig.image_path)
        st.image(img, width="stretch")
    except Exception:
        st.warning(f"Could not load image: {fig.image_path}")
        return

    import html as html_module

    # Metadata
    caption_text = fig.caption or "No caption"
    is_truncated = len(caption_text) > 80
    if is_truncated:
        truncated_part = caption_text[:80]
        safe_truncated = html_module.escape(truncated_part)
        safe_full = html_module.escape(caption_text)
        caption_html_content = (
            '<details class="fig-expandable">'
            '  <summary>'
            '    <span class="closed-view">' + safe_truncated + '<span class="fig-more-btn">...More</span></span>'
            '    <span class="open-view">' + safe_full + '<span class="fig-less-btn">...Less</span></span>'
            '  </summary>'
            '</details>'
        )
    else:
        caption_html_content = html_module.escape(caption_text)

    bbox_str = ""
    if fig.bbox:
        try:
            bbox_str = f"({fig.bbox.l:.0f}, {fig.bbox.t:.0f}, {fig.bbox.r:.0f}, {fig.bbox.b:.0f})"
        except (AttributeError, TypeError):
            bbox_str = str(fig.bbox)

    classification_str = ""
    if hasattr(fig, "classification") and fig.classification:
        conf_suffix = f" - {fig.confidence*100:.1f}%" if fig.confidence else ""
        classification_str = f" &bull; <span style='color:#fbbf24; font-weight:600;'>{fig.classification.title()}{conf_suffix}</span>"

    st.markdown(
        f"""
        <div style="
            background: var(--bg-card, #1a1a22);
            border: 1px solid var(--border-color, rgba(255,255,255,0.06));
            border-radius: 12px;
            padding: 12px 14px;
            margin-top: -0.5rem;
            margin-bottom: 1rem;
        ">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="
                    font-size:0.72rem; text-transform:uppercase;
                    letter-spacing:0.06em; color:#5c5c6e;
                ">Figure {fig.index}{classification_str}</span>
                <span style="
                    font-size:0.72rem; color:#34d399;
                    font-family:'JetBrains Mono',monospace;
                ">Page {fig.page_no}</span>
            </div>
            <p style="
                font-size:0.82rem; color:#9898a6;
                margin:0 0 4px 0; line-height:1.4;
            ">{caption_html_content}</p>
            <p style="
                font-size:0.7rem; color:#3d3d4a;
                font-family:'JetBrains Mono',monospace;
                margin:0;
            ">bbox: {bbox_str}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # VLM Description (expandable)
    if description and description != "(GPU not available — skipped)":
        with st.expander(f"VLM Description", expanded=False):
            st.markdown(
                f"""
                <div style="
                    font-size:0.85rem; color:#9898a6;
                    line-height:1.6; padding:0.5rem 0;
                ">
                    {description}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Download button
    try:
        with open(fig.image_path, "rb") as f:
            img_bytes = f.read()
        st.download_button(
            label=f"Download",
            data=img_bytes,
            file_name=Path(fig.image_path).name,
            mime="image/png",
            key=f"dl_fig_{fig.index}",
        )
    except Exception:
        pass

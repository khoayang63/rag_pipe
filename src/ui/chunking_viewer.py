"""
Chunking Viewer UI component.

Provides two modes:
1. Single Method View: Select a chunking method, view chunk cards
2. Compare Mode: Run all 3 methods and compare results side-by-side
"""

import streamlit as st
from pipeline.chunker import (
    run_chunking,
    run_all_chunking,
    CHUNKER_INFO,
    ChunkResult,
)


def _render_chunk_card(chunk, method_key: str, show_contextualized: bool = True):
    """Render a single chunk as a styled card."""
    import html as html_module

    info = CHUNKER_INFO[method_key]
    color = info["color"]

    # Determine which text to display
    display_text = chunk.contextualized if show_contextualized else chunk.text

    # Truncate very long chunks for display
    max_display = 1500
    is_truncated = len(display_text) > max_display

    # Build heading badges
    heading_html = ""
    if chunk.headings:
        badges = "".join(
            '<span class="chunk-heading-badge">' + html_module.escape(str(h)) + '</span>'
            for h in chunk.headings[:3]
        )
        heading_html = '<div class="chunk-headings">' + badges + '</div>'

    # Caption badges
    caption_html = ""
    if chunk.captions:
        cap_badges = "".join(
            '<span class="chunk-caption-badge">' + html_module.escape(str(c)[:80]) + '</span>'
            for c in chunk.captions[:2]
        )
        caption_html = '<div class="chunk-captions">' + cap_badges + '</div>'

    # Type badge
    type_html = ""
    if chunk.chunk_type:
        type_label = chunk.chunk_type.split(".")[-1] if "." in chunk.chunk_type else chunk.chunk_type
        type_html = (
            '<span class="chunk-type-badge" style="border-color:' + color + '33;color:' + color + ';">'
            + html_module.escape(type_label) + '</span>'
        )

    # Format text content with expandable details if truncated
    if is_truncated:
        truncated_part = display_text[:max_display]
        safe_truncated = html_module.escape(truncated_part).replace("\n", "<br>")
        safe_full = html_module.escape(display_text).replace("\n", "<br>")
        text_content_html = (
            '<details class="chunk-expandable">'
            '  <summary>'
            '    <span class="closed-view">' + safe_truncated + '<span class="chunk-more-btn">...More</span></span>'
            '    <span class="open-view">' + safe_full + '<span class="chunk-less-btn">...Less</span></span>'
            '  </summary>'
            '</details>'
        )
    else:
        safe_text = html_module.escape(display_text).replace("\n", "<br>")
        text_content_html = safe_text

    # Build HTML via concatenation to avoid f-string issues with { } in text
    card_html = (
        '<div class="chunk-card" style="border-left-color:' + color + ';">'
        '  <div class="chunk-card-header">'
        '    <div class="chunk-card-index" style="background:' + color + '22;color:' + color + ';">'
        '      #' + str(chunk.index)
        + '  </div>'
        '    <div class="chunk-card-tokens">'
        '      <span class="chunk-token-count">' + str(chunk.num_tokens) + '</span> tokens'
        '    </div>'
        + type_html
        + '</div>'
        + heading_html
        + caption_html
        + '<div class="chunk-card-text">' + text_content_html + '</div>'
        '</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)


def _render_stats_bar(result: ChunkResult, method_key: str):
    """Render statistics bar for a chunking result."""
    info = CHUNKER_INFO[method_key]
    color = info["color"]

    st.markdown(
        f"""
        <div class="chunk-stats-bar" style="border-color:{color}33;">
            <div class="chunk-stat-item">
                <span class="chunk-stat-value" style="color:{color};">{result.num_chunks}</span>
                <span class="chunk-stat-label">Chunks</span>
            </div>
            <div class="chunk-stat-item">
                <span class="chunk-stat-value" style="color:{color};">{result.total_tokens:,}</span>
                <span class="chunk-stat-label">Total Tokens</span>
            </div>
            <div class="chunk-stat-item">
                <span class="chunk-stat-value" style="color:{color};">{result.avg_tokens_per_chunk}</span>
                <span class="chunk-stat-label">Avg Tokens/Chunk</span>
            </div>
            <div class="chunk-stat-item">
                <span class="chunk-stat-value" style="color:{color};">{result.min_tokens}</span>
                <span class="chunk-stat-label">Min Tokens</span>
            </div>
            <div class="chunk-stat-item">
                <span class="chunk-stat-value" style="color:{color};">{result.max_tokens}</span>
                <span class="chunk-stat-label">Max Tokens</span>
            </div>
            <div class="chunk-stat-item">
                <span class="chunk-stat-value" style="color:{color};">{result.chunking_time:.3f}s</span>
                <span class="chunk-stat-label">Time</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_comparison_table(all_results: dict[str, ChunkResult]):
    """Render a comparison table across all chunking methods."""
    rows_html = ""
    for method_key, result in all_results.items():
        info = CHUNKER_INFO[method_key]
        color = info["color"]
        icon = info["icon"]
        label = info["label"]

        rows_html += (
            '<tr>'
            f'<td style="color:{color}; font-weight:600;">{icon} {label}</td>'
            f'<td style="text-align:center;">{result.num_chunks}</td>'
            f'<td style="text-align:center;">{result.total_tokens:,}</td>'
            f'<td style="text-align:center;">{result.avg_tokens_per_chunk}</td>'
            f'<td style="text-align:center;">{result.min_tokens}</td>'
            f'<td style="text-align:center;">{result.max_tokens}</td>'
            f'<td style="text-align:center;">{result.chunking_time:.3f}s</td>'
            '</tr>'
        )

    html_content = (
        '<div class="chunk-compare-table-wrap">'
        '<table class="chunk-compare-table">'
        '<thead>'
        '<tr>'
        '<th>Method</th>'
        '<th style="text-align:center;">Chunks</th>'
        '<th style="text-align:center;">Total Tokens</th>'
        '<th style="text-align:center;">Avg Tokens</th>'
        '<th style="text-align:center;">Min</th>'
        '<th style="text-align:center;">Max</th>'
        '<th style="text-align:center;">Time</th>'
        '</tr>'
        '</thead>'
        '<tbody>'
        + rows_html +
        '</tbody>'
        '</table>'
        '</div>'
    )
    st.markdown(html_content, unsafe_allow_html=True)


def _render_comparison_bars(all_results: dict[str, ChunkResult]):
    """Render horizontal bar chart comparing chunk counts and avg tokens."""
    max_chunks = max(r.num_chunks for r in all_results.values()) or 1
    max_avg = max(r.avg_tokens_per_chunk for r in all_results.values()) or 1

    bars_html = ""
    for method_key, result in all_results.items():
        info = CHUNKER_INFO[method_key]
        color = info["color"]
        icon = info["icon"]

        chunk_pct = (result.num_chunks / max_chunks) * 100
        avg_pct = (result.avg_tokens_per_chunk / max_avg) * 100

        bars_html += (
            '<div class="chunk-bar-group">'
            f'<div class="chunk-bar-label" style="color:{color};">{icon} {info["label"].split(" (")[0]}</div>'
            '<div class="chunk-bar-row">'
            '<span class="chunk-bar-metric">Chunks</span>'
            '<div class="chunk-bar-track">'
            f'<div class="chunk-bar-fill" style="width:{chunk_pct}%;background:{color};"></div>'
            '</div>'
            f'<span class="chunk-bar-value">{result.num_chunks}</span>'
            '</div>'
            '<div class="chunk-bar-row">'
            '<span class="chunk-bar-metric">Avg Tok</span>'
            '<div class="chunk-bar-track">'
            f'<div class="chunk-bar-fill" style="width:{avg_pct}%;background:{color}99;"></div>'
            '</div>'
            f'<span class="chunk-bar-value">{result.avg_tokens_per_chunk}</span>'
            '</div>'
            '</div>'
        )

    html_content = (
        '<div class="chunk-bars-container">'
        '<h4 style="margin:0 0 1rem 0; font-size:0.9rem; color:#f0f0f3;">'
        '📊 Visual Comparison'
        '</h4>'
        + bars_html +
        '</div>'
    )
    st.markdown(html_content, unsafe_allow_html=True)


def render_chunking_viewer(document, doc_id: str = "0"):
    """
    Render the chunking viewer tab for a document.

    Args:
        document: DoclingDocument from conversion result
        doc_id: Unique ID for this document (for Streamlit widget keys)
    """
    # ── Header ──
    st.markdown(
        """
        <div style="margin-bottom:1rem;">
            <h3 style="margin:0; font-size:1.1rem; color:#f0f0f3;">
                🧩 Document Chunking
            </h3>
            <p style="margin:4px 0 0 0; font-size:0.78rem; color:#5c5c6e;">
                Split the document into chunks for RAG embedding. Uses <code>chunker.contextualize()</code> for enriched output.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Controls ──
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 1.5, 1, 1.5])

    with ctrl_col1:
        method_options = list(CHUNKER_INFO.keys())
        method_labels = [CHUNKER_INFO[m]["label"] for m in method_options]
        selected_idx = st.selectbox(
            "Chunking Method",
            options=range(len(method_options)),
            format_func=lambda i: f"{CHUNKER_INFO[method_options[i]]['icon']} {method_labels[i]}",
            index=1,  # default to hybrid
            key=f"chunk_method_{doc_id}",
        )
        selected_method = method_options[selected_idx]

    with ctrl_col2:
        info = CHUNKER_INFO[selected_method]
        max_tokens = 512
        if info["supports_max_tokens"]:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=64,
                max_value=4096,
                value=512,
                step=64,
                key=f"chunk_max_tokens_{doc_id}",
            )
        else:
            st.markdown(
                '<p style="font-size:0.75rem; color:#5c5c6e; margin-top:2rem;">Max tokens N/A for this method</p>',
                unsafe_allow_html=True,
            )

    with ctrl_col3:
        merge_peers = True
        if info["supports_merge_peers"]:
            merge_peers = st.toggle(
                "Merge Peers",
                value=True,
                key=f"chunk_merge_{doc_id}",
            )
        else:
            st.markdown(
                '<p style="font-size:0.75rem; color:#5c5c6e; margin-top:2rem;">Merge N/A</p>',
                unsafe_allow_html=True,
            )

    with ctrl_col4:
        compare_mode = st.toggle(
            "🔀 Compare All",
            value=False,
            key=f"chunk_compare_{doc_id}",
            help="Run all 3 chunking methods and compare results side-by-side",
        )

    # ── Method description ──
    if not compare_mode:
        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                        border-radius:8px; padding:10px 14px; margin:0.5rem 0 1rem 0;">
                <span style="font-size:0.82rem; color:#9898a6;">
                    {CHUNKER_INFO[selected_method]['icon']} {CHUNKER_INFO[selected_method]['description']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Run button ──
    run_label = "Run All Chunkers" if compare_mode else f"Run {CHUNKER_INFO[selected_method]['label'].split(' (')[0]}"
    run_clicked = st.button(
        f"🔄 {run_label}",
        key=f"chunk_run_{doc_id}",
        type="primary",
        use_container_width=False,
    )

    # ── Session state for cached results ──
    result_key = f"chunk_results_{doc_id}"
    compare_key = f"chunk_compare_results_{doc_id}"

    if run_clicked:
        if compare_mode:
            with st.status("Running all 3 chunking methods...", expanded=True) as status:
                st.write("Running HierarchicalChunker...")
                all_results = {}
                for m in ["hierarchical", "hybrid", "line_based"]:
                    st.write(f"Running {CHUNKER_INFO[m]['label']}...")
                    all_results[m] = run_chunking(
                        document=document,
                        method=m,
                        max_tokens=max_tokens,
                        merge_peers=merge_peers,
                    )
                st.session_state[compare_key] = all_results
                st.session_state[result_key] = None  # clear single
                total_chunks = sum(r.num_chunks for r in all_results.values())
                status.update(
                    label=f"Compare complete — {total_chunks} total chunks",
                    state="complete",
                )
        else:
            with st.status(f"Running {CHUNKER_INFO[selected_method]['label']}...", expanded=True) as status:
                result = run_chunking(
                    document=document,
                    method=selected_method,
                    max_tokens=max_tokens,
                    merge_peers=merge_peers,
                )
                st.session_state[result_key] = (selected_method, result)
                st.session_state[compare_key] = None  # clear compare
                status.update(
                    label=f"Done — {result.num_chunks} chunks in {result.chunking_time:.3f}s",
                    state="complete",
                )

    # ── Display results ──
    if compare_mode and st.session_state.get(compare_key):
        all_results = st.session_state[compare_key]

        # Comparison table
        _render_comparison_table(all_results)

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        # Visual bar comparison
        _render_comparison_bars(all_results)

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        # Show contextualized toggle
        show_ctx = st.toggle(
            "Show contextualized text",
            value=True,
            key=f"chunk_ctx_compare_{doc_id}",
            help="Show text enriched with headings/captions via chunker.contextualize()",
        )

        # Side-by-side chunks
        st.markdown(
            '<h4 style="margin:1rem 0 0.5rem 0; font-size:0.9rem; color:#f0f0f3;">Chunks by Method</h4>',
            unsafe_allow_html=True,
        )
        tabs = st.tabs([
            f"{CHUNKER_INFO[m]['icon']} {CHUNKER_INFO[m]['label'].split(' (')[0]} ({r.num_chunks})"
            for m, r in all_results.items()
        ])
        for tab, (method_key, result) in zip(tabs, all_results.items()):
            with tab:
                _render_stats_bar(result, method_key)
                for chunk in result.chunks:
                    _render_chunk_card(chunk, method_key, show_contextualized=show_ctx)

    elif not compare_mode and st.session_state.get(result_key):
        method_key, result = st.session_state[result_key]

        # Stats bar
        _render_stats_bar(result, method_key)

        # Contextualized toggle
        show_ctx = st.toggle(
            "Show contextualized text",
            value=True,
            key=f"chunk_ctx_single_{doc_id}",
            help="Show text enriched with headings/captions via chunker.contextualize()",
        )

        # Chunk cards
        for chunk in result.chunks:
            _render_chunk_card(chunk, method_key, show_contextualized=show_ctx)

    elif not run_clicked:
        # Show placeholder
        st.markdown(
            """
            <div class="chunk-placeholder">
                <div class="chunk-placeholder-icon">🧩</div>
                <p>Select a chunking method and click <strong>Run</strong> to split this document into chunks.</p>
                <p style="font-size:0.75rem; color:#5c5c6e;">
                    Enable <strong>Compare All</strong> to run all 3 methods and see side-by-side comparison.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

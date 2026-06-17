"""
Spell Correction Viewer UI component.

Provides chunk-by-chunk Vietnamese spell correction using
bmd1905/vietnamese-correction-v2 with side-by-side diff highlighting.
Each chunk has its own "Correct" and "Edit manually" options, allowing
users to clean up spelling and OCR layout errors prior to DB ingestion.
"""

import html as html_module
import re
import streamlit as st


def _render_diff_html(original: str, corrected: str, diffs) -> tuple[str, str]:
    """
    Render original and corrected text with inline diff highlighting.

    Returns (original_html, corrected_html) with highlighted spans.
    """
    if not diffs:
        safe = html_module.escape(original).replace("\n", "<br>")
        return safe, safe

    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, original, corrected)

    orig_parts = []
    corr_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        orig_chunk = html_module.escape(original[i1:i2]).replace("\n", "<br>")
        corr_chunk = html_module.escape(corrected[j1:j2]).replace("\n", "<br>")

        if tag == "equal":
            orig_parts.append(orig_chunk)
            corr_parts.append(corr_chunk)
        elif tag == "replace":
            orig_parts.append(
                f'<span style="background:rgba(239,68,68,0.25); color:#fca5a5; '
                f'text-decoration:line-through; padding:1px 3px; border-radius:3px;">'
                f'{orig_chunk}</span>'
            )
            corr_parts.append(
                f'<span style="background:rgba(52,211,153,0.25); color:#6ee7b7; '
                f'padding:1px 3px; border-radius:3px; font-weight:600;">'
                f'{corr_chunk}</span>'
            )
        elif tag == "delete":
            orig_parts.append(
                f'<span style="background:rgba(239,68,68,0.25); color:#fca5a5; '
                f'text-decoration:line-through; padding:1px 3px; border-radius:3px;">'
                f'{orig_chunk}</span>'
            )
        elif tag == "insert":
            corr_parts.append(
                f'<span style="background:rgba(52,211,153,0.25); color:#6ee7b7; '
                f'padding:1px 3px; border-radius:3px; font-weight:600;">'
                f'{corr_chunk}</span>'
            )

    return "".join(orig_parts), "".join(corr_parts)


def _render_chunk_card(para, doc_id: str, state_key: str):
    """Render a single chunk card with optional correction button and manual editor."""
    idx = para["index"]
    original = para["original"]
    is_skipped = para.get("is_skipped", False)
    correction = para.get("correction")  # ParagraphCorrection/ChunkCorrection or None
    is_applied = para.get("applied", False)

    # Determine display state
    if is_skipped:
        # Skipped chunk (markdown syntax or only numbers) — show as dimmed
        safe_text = html_module.escape(original).replace("\n", "<br>")
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04); '
            f'border-radius:8px; padding:10px 14px; margin:4px 0; opacity:0.5;">'
            f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">'
            f'<span style="font-size:0.7rem; color:#5c5c6e; font-family:\'JetBrains Mono\',monospace;">Chunk #{idx}</span>'
            f'<span style="font-size:0.65rem; color:#5c5c6e; background:rgba(255,255,255,0.05); '
            f'padding:2px 8px; border-radius:4px;">SKIPPED (syntax/non-alpha)</span>'
            f'</div>'
            f'<div style="font-size:0.8rem; color:#5c5c6e; line-height:1.5;">{safe_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    if correction and correction.has_changes:
        # Has corrections — show diff view
        orig_html, corr_html = _render_diff_html(
            correction.original, correction.corrected, correction.diffs
        )

        # Badge showing number of changes
        num_changes = len(correction.diffs)
        badge_color = "#fbbf24" if num_changes <= 3 else "#f87171"

        # Applied status
        if is_applied:
            status_badge = (
                '<span style="font-size:0.65rem; color:#34d399; background:rgba(52,211,153,0.1); '
                'padding:2px 8px; border-radius:4px; border:1px solid rgba(52,211,153,0.2);">✓ APPLIED</span>'
            )
        else:
            status_badge = ""

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(251,191,36,0.15); '
            f'border-left:3px solid {badge_color}; border-radius:8px; padding:12px 16px; margin:6px 0;">'
            f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">'
            f'<span style="font-size:0.7rem; color:#9898a6; font-family:\'JetBrains Mono\',monospace;">Chunk #{idx}</span>'
            f'<span style="font-size:0.65rem; color:{badge_color}; background:{badge_color}18; '
            f'padding:2px 8px; border-radius:4px; border:1px solid {badge_color}33;">'
            f'{num_changes} change{"s" if num_changes > 1 else ""}</span>'
            f'{status_badge}'
            f'</div>'
            f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">'
            # Original column
            f'<div>'
            f'<div style="font-size:0.68rem; color:#f87171; text-transform:uppercase; '
            f'letter-spacing:0.05em; margin-bottom:6px; font-weight:600;">Original</div>'
            f'<div style="font-size:0.82rem; color:#e4e4e7; line-height:1.65; '
            f'background:rgba(239,68,68,0.04); border:1px solid rgba(239,68,68,0.08); '
            f'border-radius:6px; padding:10px 12px;">{orig_html}</div>'
            f'</div>'
            # Corrected column
            f'<div>'
            f'<div style="font-size:0.68rem; color:#34d399; text-transform:uppercase; '
            f'letter-spacing:0.05em; margin-bottom:6px; font-weight:600;">Corrected</div>'
            f'<div style="font-size:0.82rem; color:#e4e4e7; line-height:1.65; '
            f'background:rgba(52,211,153,0.04); border:1px solid rgba(52,211,153,0.08); '
            f'border-radius:6px; padding:10px 12px;">{corr_html}</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Apply button (only if not already applied)
        if not is_applied:
            if st.button(
                f"✅ Apply correction #{idx}",
                key=f"apply_correction_{state_key}_{idx}",
                use_container_width=False,
            ):
                # Store as applied
                if state_key in st.session_state:
                    paras = st.session_state[state_key]
                    for p in paras:
                        if p["index"] == idx:
                            p["applied"] = True
                            p["current_text"] = correction.corrected
                            break
                st.rerun()

    elif correction and not correction.has_changes:
        # Corrected but no changes found
        max_display = 1200
        safe_text = html_module.escape(original).replace("\n", "<br>")
        if len(original) > max_display:
            first_part = html_module.escape(original[:max_display]).replace("\n", "<br>")
            second_part = html_module.escape(original[max_display:]).replace("\n", "<br>")
            toggle_id = f"read-more-toggle-{state_key}-{idx}".replace("_", "-")
            display_html = (
                f'<input type="checkbox" class="read-more-state" id="{toggle_id}">'
                f'<span class="read-more-wrap">'
                f'{first_part}'
                f'<label for="{toggle_id}" class="read-more-trigger read-more-trigger-more">... [more]</label>'
                f'<span class="read-more-target">{second_part}</span>'
                f'<label for="{toggle_id}" class="read-more-trigger read-more-trigger-less"> [less]</label>'
                f'</span>'
            )
        else:
            display_html = safe_text

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); '
            f'border-radius:8px; padding:10px 14px; margin:4px 0;">'
            f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">'
            f'<span style="font-size:0.7rem; color:#9898a6; font-family:\'JetBrains Mono\',monospace;">Chunk #{idx}</span>'
            f'<span style="font-size:0.65rem; color:#34d399; background:rgba(52,211,153,0.1); '
            f'padding:2px 8px; border-radius:4px;">✓ No errors</span>'
            f'</div>'
            f'<div style="font-size:0.82rem; color:#c0c0cc; line-height:1.5;">{display_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        # Not yet corrected — show original with a correct button
        max_display = 1200
        safe_text = html_module.escape(original).replace("\n", "<br>")
        if len(original) > max_display:
            first_part = html_module.escape(original[:max_display]).replace("\n", "<br>")
            second_part = html_module.escape(original[max_display:]).replace("\n", "<br>")
            toggle_id = f"read-more-toggle-{state_key}-{idx}".replace("_", "-")
            display_html = (
                f'<input type="checkbox" class="read-more-state" id="{toggle_id}">'
                f'<span class="read-more-wrap">'
                f'{first_part}'
                f'<label for="{toggle_id}" class="read-more-trigger read-more-trigger-more">... [more]</label>'
                f'<span class="read-more-target">{second_part}</span>'
                f'<label for="{toggle_id}" class="read-more-trigger read-more-trigger-less"> [less]</label>'
                f'</span>'
            )
        else:
            display_html = safe_text

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); '
            f'border-radius:8px; padding:10px 14px; margin:4px 0;">'
            f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">'
            f'<span style="font-size:0.7rem; color:#9898a6; font-family:\'JetBrains Mono\',monospace;">Chunk #{idx}</span>'
            f'</div>'
            f'<div style="font-size:0.82rem; color:#c0c0cc; line-height:1.5;">{display_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button(
            f"🔍 Correct #{idx}",
            key=f"correct_para_{state_key}_{idx}",
            use_container_width=False,
        ):
            from pipeline.processing.spell_corrector import get_spell_corrector

            with st.spinner(f"Correcting chunk {idx}..."):
                corrector = get_spell_corrector()
                result = corrector.correct_paragraph(original, idx)

            # Store correction result
            if state_key in st.session_state:
                paras = st.session_state[state_key]
                for p in paras:
                    if p["index"] == idx:
                        p["correction"] = result
                        if not result.has_changes:
                            p["current_text"] = original
                        break
            st.rerun()

    # Manual Edit section for any non-skipped chunk
    if not is_skipped:
        with st.expander("✏️ Edit manually", expanded=False):
            current_val = para.get("current_text", original)
            edited_text = st.text_area(
                "Modify chunk text:",
                value=current_val,
                key=f"edit_text_{state_key}_{idx}",
                height=120,
            )
            if st.button("💾 Save & Apply Edit", key=f"save_edit_{state_key}_{idx}"):
                from pipeline.processing.spell_corrector import ParagraphCorrection, _compute_diffs
                if state_key in st.session_state:
                    paras = st.session_state[state_key]
                    for p in paras:
                        if p["index"] == idx:
                            p["applied"] = True
                            p["current_text"] = edited_text
                            
                            # Create a correction object representing the manual changes
                            diffs = _compute_diffs(original, edited_text)
                            p["correction"] = ParagraphCorrection(
                                index=idx,
                                original=original,
                                corrected=edited_text,
                                diffs=diffs,
                                has_changes=len(diffs) > 0,
                                is_skipped=False
                            )
                            break
                st.rerun()


def render_spell_correction_viewer(doc_id: str = "0"):
    """
    Render the spell correction viewer tab (Chunk-based).

    Loads generated chunks from memory, allows selective correction of each chunk
    using the fine-tuned Seq2Seq model, and provides manual edit capabilities.
    """
    # ── Header & Inline Style ──
    st.markdown(
        """
        <style>
        input.read-more-state {
            display: none;
        }
        label.read-more-trigger {
            color: #60a5fa;
            font-weight: 600;
            cursor: pointer;
            display: inline;
            user-select: none;
        }
        label.read-more-trigger:hover {
            color: #93c5fd;
            text-decoration: underline;
        }
        .read-more-wrap .read-more-target,
        .read-more-wrap .read-more-trigger-less {
            display: none;
        }
        input.read-more-state:checked ~ .read-more-wrap .read-more-target,
        input.read-more-state:checked ~ .read-more-wrap .read-more-trigger-less {
            display: inline;
        }
        input.read-more-state:checked ~ .read-more-wrap .read-more-trigger-more {
            display: none;
        }
        </style>
        <div style="margin-bottom:1rem;">
            <h3 style="margin:0; font-size:1.1rem; color:#f0f0f3;">
                ✏️ Vietnamese Spell Correction (Chunk-based)
            </h3>
            <p style="margin:4px 0 0 0; font-size:0.78rem; color:#5c5c6e;">
                Correct spelling, OCR, and word layout errors on generated Chunks using
                <code>bmd1905/vietnamese-correction-v2</code>. Corrected chunks are ingested into pgvector.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Detect Available Chunk results ──
    result_key = f"chunk_results_{doc_id}"
    compare_key = f"chunk_compare_results_{doc_id}"

    available_methods = []
    
    # Check single run results
    if st.session_state.get(result_key):
        method_key, result = st.session_state[result_key]
        available_methods.append((f"single_{method_key}", f"Current Run: {result.method_label} ({result.num_chunks} chunks)"))
        
    # Check compare run results
    if st.session_state.get(compare_key):
        compare_results = st.session_state[compare_key]
        for m, r in compare_results.items():
            available_methods.append((f"compare_{m}", f"Compare Mode: {r.method_label} ({r.num_chunks} chunks)"))

    if not available_methods:
        st.markdown(
            """
            <div class="chunk-placeholder" style="margin-top: 1.5rem; text-align: center; padding: 2.5rem 1.5rem; background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.08); border-radius: 8px;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🧩</div>
                <h4 style="margin: 0 0 0.5rem 0; color: #f0f0f3; font-size: 0.95rem;">Chưa tìm thấy dữ liệu Chunking</h4>
                <p style="margin: 0; font-size: 0.8rem; color: #5c5c6e; max-width: 500px; margin: 0 auto;">
                    Vui lòng di chuyển đến tab <strong>Chunking</strong> và chạy bộ chia nhỏ tài liệu (Chunker) trước khi tiến hành sửa lỗi chính tả.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Select box to choose which chunk method to correct
    selected_source_idx = st.selectbox(
        "Select Chunking Output to Correct",
        options=range(len(available_methods)),
        format_func=lambda i: available_methods[i][1],
        key=f"spell_source_select_{doc_id}"
    )
    
    selected_source_key = available_methods[selected_source_idx][0]

    # Get active chunks based on selection
    active_chunks = []
    if selected_source_key.startswith("single_"):
        _, result = st.session_state[result_key]
        active_chunks = result.chunks
    else:
        method_m = selected_source_key.split("compare_")[-1]
        active_chunks = st.session_state[compare_key][method_m].chunks

    # ── Initialize/Synchronize chunk correction state ──
    state_key = f"spell_chunks_{doc_id}_{selected_source_key}"
    
    # Re-initialize if length mismatch occurs (e.g. user re-ran chunker with different token size)
    needs_init = (state_key not in st.session_state) or (len(st.session_state[state_key]) != len(active_chunks))
    
    if needs_init:
        from pipeline.processing.spell_corrector import should_skip_chunk

        st.session_state[state_key] = [
            {
                "index": c.index,
                "original": c.contextualized,
                "current_text": c.contextualized,
                "correction": None,
                "applied": False,
                "is_skipped": should_skip_chunk(c.contextualized),
            }
            for c in active_chunks
        ]

    paras = st.session_state[state_key]

    # ── Statistics ──
    total = len(paras)
    corrected_count = sum(1 for p in paras if p["correction"] is not None)
    changed_count = sum(
        1 for p in paras
        if p["correction"] is not None and p["correction"].has_changes
    )
    applied_count = sum(1 for p in paras if p["applied"])
    skipped_count = sum(1 for p in paras if p["is_skipped"])
    correctable = total - skipped_count

    st.markdown(
        f"""
        <div style="display:flex; gap:1.5rem; flex-wrap:wrap; align-items:center;
                    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                    border-radius:8px; padding:10px 16px; margin-bottom:1rem; margin-top: 0.5rem;">
            <div style="text-align:center;">
                <div style="font-size:1.2rem; font-weight:700; color:#a78bfa;
                     font-family:'JetBrains Mono',monospace;">{total}</div>
                <div style="font-size:0.68rem; color:#5c5c6e; text-transform:uppercase;
                     letter-spacing:0.05em;">Chunks</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.2rem; font-weight:700; color:#60a5fa;
                     font-family:'JetBrains Mono',monospace;">{corrected_count}/{correctable}</div>
                <div style="font-size:0.68rem; color:#5c5c6e; text-transform:uppercase;
                     letter-spacing:0.05em;">Checked</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.2rem; font-weight:700; color:#fbbf24;
                     font-family:'JetBrains Mono',monospace;">{changed_count}</div>
                <div style="font-size:0.68rem; color:#5c5c6e; text-transform:uppercase;
                     letter-spacing:0.05em;">With Errors</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.2rem; font-weight:700; color:#34d399;
                     font-family:'JetBrains Mono',monospace;">{applied_count}</div>
                <div style="font-size:0.68rem; color:#5c5c6e; text-transform:uppercase;
                     letter-spacing:0.05em;">Applied</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.2rem; font-weight:700; color:#5c5c6e;
                     font-family:'JetBrains Mono',monospace;">{skipped_count}</div>
                <div style="font-size:0.68rem; color:#5c5c6e; text-transform:uppercase;
                     letter-spacing:0.05em;">Skipped</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Bulk controls ──
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

    with ctrl_col1:
        correct_all = st.button(
            "🔍 Correct All Chunks",
            key=f"correct_all_{state_key}",
            type="primary",
            use_container_width=True,
        )

    with ctrl_col2:
        apply_all = st.button(
            "✅ Apply All Corrections",
            key=f"apply_all_{state_key}",
            use_container_width=True,
            disabled=(changed_count == 0),
        )

    with ctrl_col3:
        if applied_count > 0:
            st.markdown(
                f'<div style="display:flex; align-items:center; height:100%; padding-top:6px;">'
                f'<span style="font-size:0.78rem; color:#34d399;">✓ {applied_count} correction(s) applied — '
                f'corrected text will be used during pgvector DB ingestion</span></div>',
                unsafe_allow_html=True,
            )

    # ── Correct All logic ──
    if correct_all:
        from pipeline.processing.spell_corrector import get_spell_corrector

        with st.status("Correcting all chunks...", expanded=True) as status:
            corrector = get_spell_corrector()
            for p in paras:
                if p["is_skipped"] or p["correction"] is not None:
                    continue
                st.write(f"Correcting Chunk #{p['index']}...")
                result = corrector.correct_paragraph(p["original"], p["index"])
                p["correction"] = result
                if not result.has_changes:
                    p["current_text"] = p["original"]

            total_checked = sum(1 for p in paras if p["correction"] is not None)
            total_changes = sum(
                1 for p in paras
                if p["correction"] is not None and p["correction"].has_changes
            )
            status.update(
                label=f"Done — {total_checked} chunks checked, {total_changes} with errors",
                state="complete",
            )
        st.rerun()

    # ── Apply All logic ──
    if apply_all:
        for p in paras:
            if p["correction"] and p["correction"].has_changes and not p["applied"]:
                p["applied"] = True
                p["current_text"] = p["correction"].corrected
        st.rerun()

    # ── Filter controls ──
    filter_options = ["All", "With Errors", "Applied", "Unchecked"]
    selected_filter = st.radio(
        "Show",
        options=filter_options,
        horizontal=True,
        key=f"spell_filter_{state_key}",
        label_visibility="collapsed",
    )

    # ── Chunk cards ──
    for para in paras:
        # Apply filter
        if selected_filter == "With Errors":
            if not (para["correction"] and para["correction"].has_changes):
                continue
        elif selected_filter == "Applied":
            if not para["applied"]:
                continue
        elif selected_filter == "Unchecked":
            if para["correction"] is not None or para["is_skipped"]:
                continue

        _render_chunk_card(para, doc_id, state_key)

    # ── Export section ──
    if applied_count > 0:
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        # Reassemble chunks into full text
        corrected_md = "\n\n".join([p["current_text"] for p in paras])

        with st.expander("📄 View Reassembled Corrected Content", expanded=False):
            st.code(corrected_md, language="markdown", line_numbers=True)
            st.download_button(
                label="Download reassembled .md",
                data=corrected_md,
                file_name="corrected_chunks.md",
                mime="text/markdown",
                key=f"download_corrected_{state_key}",
            )

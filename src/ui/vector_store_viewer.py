"""
Vector Store UI Tab component.
Handles PostgreSQL connection verification, chunk ingestion with BgeEmbedder, and similarity search comparison.
"""

import streamlit as st
import time
import uuid
import html as html_module
from pipeline.indexer.postgres_store import VectorStore
from pipeline.processing.embedder import get_embedder

def render_vector_store_viewer(doc_name: str, doc_id: str = "0"):
    """
    Render the Vector DB ingestion and search tab.
    
    Args:
        doc_name: The name of the currently active document
        doc_id: Unique ID for the document (for Streamlit keys)
    """
    db = VectorStore()
    
    # ── Title ──
    st.markdown(
        """
        <div style="margin-bottom:1rem;">
            <h3 style="margin:0; font-size:1.1rem; color:#f0f0f3;">
                🗄️ pgvector Ingestion & Search
            </h3>
            <p style="margin:4px 0 0 0; font-size:0.78rem; color:#5c5c6e;">
                Generate dense embeddings using <code>BAAI/bge-m3</code>, store chunks in PostgreSQL, and search using hybrid retrieval.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Database Connection Status Check ──
    conn_info = db.test_connection()
    
    if not conn_info["connected"]:
        st.markdown(
            f"""
            <div class="glass-card animate-in" style="border-left: 4px solid var(--danger, #ef4444); margin-bottom: 1.5rem;">
                <h4 style="margin:0 0 0.5rem 0; color:#ef4444; font-size:0.9rem;">⚠️ PostgreSQL Database Disconnected</h4>
                <p style="margin:0 0 0.8rem 0; font-size:0.8rem; color:#9898a6; line-height:1.5;">
                    Could not connect to PostgreSQL. Please ensure that Docker Desktop is running and you have started the database containers.
                </p>
                <div style="background:rgba(0,0,0,0.2); padding:10px 12px; border-radius:6px; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#f0f0f3; border:1px solid rgba(255,255,255,0.04);">
                    docker compose up -d
                </div>
                <p style="margin:0.5rem 0 0 0; font-size:0.72rem; color:#5c5c6e;">
                    Connection details: <code>{db.conn_params['host']}:{db.conn_params['port']}</code> | User: <code>{db.conn_params['user']}</code> | Error: <code>{conn_info['error']}</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Database is connected, show status
    pgvector_status = "Installed" if conn_info["pgvector_installed"] else "Missing"
    pgvector_color = "#34d399" if conn_info["pgvector_installed"] else "#ef4444"
    
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; gap:12px; margin-bottom:1.5rem;">
            <div class="chunk-stats-bar" style="border-color:rgba(52,211,153,0.15); flex:1; min-width:200px;">
                <div class="chunk-stat-item">
                    <span class="chunk-stat-value" style="color:#34d399;">Connected</span>
                    <span class="chunk-stat-label">DB Status</span>
                </div>
                <div class="chunk-stat-item">
                    <span class="chunk-stat-value" style="color:{pgvector_color};">{pgvector_status}</span>
                    <span class="chunk-stat-label">pgvector Extension</span>
                </div>
                <div class="chunk-stat-item">
                    <span class="chunk-stat-value" style="color:#60a5fa;">{db.conn_params['database']}</span>
                    <span class="chunk-stat-label">Active DB</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── UI layout: Left for Ingestion, Right for Search ──
    col_ingest, col_search = st.columns([1.2, 1.8])

    # ── Left Column: Chunks Ingestion ──
    with col_ingest:
        st.markdown('<h4 style="margin:0 0 0.8rem 0; font-size:0.9rem; color:#f0f0f3;">🚀 Ingest Document Chunks</h4>', unsafe_allow_html=True)
        
        # Check what chunk results are cached in Streamlit session state
        result_key = f"chunk_results_{doc_id}"
        compare_key = f"chunk_compare_results_{doc_id}"
        
        chunk_source = None
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
                <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:12px; text-align:center;">
                    <p style="margin:0; font-size:0.78rem; color:#5c5c6e;">No chunk results found in memory.</p>
                    <p style="margin:4px 0 0 0; font-size:0.72rem; color:#3d3d4a;">Please go to the <strong>Document Chunking</strong> tab and run a chunker first.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            selected_source_idx = st.selectbox(
                "Select Chunking Output to Ingest",
                options=range(len(available_methods)),
                format_func=lambda i: available_methods[i][1],
                key=f"ingest_source_select_{doc_id}"
            )
            
            selected_source_key = available_methods[selected_source_idx][0]
            
            # Extract active chunk items to ingest
            active_chunks = []
            if selected_source_key.startswith("single_"):
                _, result = st.session_state[result_key]
                active_chunks = result.chunks
            else:
                method_m = selected_source_key.split("compare_")[-1]
                active_chunks = st.session_state[compare_key][method_m].chunks

            # Check if corrected chunks exist in session state
            spell_key = f"spell_chunks_{doc_id}_{selected_source_key}"
            corrected_lookup = {}
            applied_count = 0
            if spell_key in st.session_state:
                spell_data = st.session_state[spell_key]
                corrected_lookup = {
                    item["index"]: item["current_text"]
                    for item in spell_data
                }
                applied_count = sum(1 for item in spell_data if item["applied"])

            # Status HTML badge for spelling correction status
            if applied_count > 0:
                spell_status_html = f'<span style="font-size:0.72rem; color:#34d399; font-weight:600;">✓ {applied_count} correction(s) applied (will ingest corrected text)</span>'
            else:
                spell_status_html = '<span style="font-size:0.72rem; color:#5c5c6e;">No corrections applied (ingesting original text)</span>'

            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04); border-radius:8px; padding:10px 12px; margin-bottom:1rem;">
                    <span style="font-size:0.75rem; color:#9898a6;">Chunks to index: <strong>{len(active_chunks)}</strong></span><br>
                    <span style="font-size:0.72rem; color:#5c5c6e;">Model: <code>BAAI/bge-m3</code> (1024-dim dense vectors)</span><br>
                    <div style="margin-top: 4px;">{spell_status_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            ingest_btn = st.button("🚀 Ingest into Postgres", key=f"ingest_btn_{doc_id}", use_container_width=True)
            
            if ingest_btn:
                try:
                    # Inform user about potential first-time model download
                    info_placeholder = st.empty()
                    info_placeholder.info(
                        "ℹ️ **First-time Run:** Loading the `BAAI/bge-m3` model. If it is not cached, "
                        "this will download about **2.2 GB** of weights. "
                        "Please check the terminal console where you started Streamlit to monitor the download progress bar."
                    )
                    
                    with st.spinner("Loading BAAI/bge-m3 embedding model (this may take a few minutes if downloading)..."):
                        embedder = get_embedder()
                    
                    info_placeholder.empty()
                    
                    progress_text = "Generating embeddings & indexing chunks..."
                    my_bar = st.progress(0.0, text=progress_text)
                    
                    # Package chunk data
                    payload_chunks = []
                    texts_to_embed = []
                    
                    for chunk in active_chunks:
                        # Use corrected contextualized text if available, otherwise original
                        text_to_use = corrected_lookup.get(chunk.index, chunk.contextualized)
                        
                        payload_chunks.append({
                            "index": chunk.index,
                            "text": chunk.text,
                            "contextualized": text_to_use,
                            "page_no": chunk.page_no if hasattr(chunk, 'page_no') else 1,
                            "chunk_type": chunk.chunk_type,
                            "headings": chunk.headings,
                            "captions": chunk.captions
                        })
                        # Use contextualized text for embeddings for richer retrieval semantics
                        texts_to_embed.append(text_to_use)
                    
                    # Batch embedding generation (batch size = 16 to protect VRAM/RAM)
                    batch_size = 16
                    all_embeddings = []
                    total_chunks = len(texts_to_embed)
                    
                    for i in range(0, total_chunks, batch_size):
                        batch_texts = texts_to_embed[i:i+batch_size]
                        # Update progress bar
                        progress_val = min(i / total_chunks, 1.0)
                        my_bar.progress(progress_val, text=f"Embedding chunks {i}-{min(i+batch_size, total_chunks)} of {total_chunks}...")
                        
                        batch_embs = embedder.get_embeddings(batch_texts)
                        all_embeddings.extend(batch_embs)
                    
                    # Attach embeddings to payload
                    for idx, emb in enumerate(all_embeddings):
                        payload_chunks[idx]["embedding"] = emb
                        
                    # Save to database
                    my_bar.progress(0.9, text="Saving to Postgres (pgvector)...")
                    
                    # Use unique document id from conversion result or generate one
                    db_doc_id = doc_id
                    ingested_count = db.ingest_document(
                        doc_id=db_doc_id,
                        doc_name=doc_name,
                        chunks=payload_chunks
                    )
                    
                    my_bar.progress(1.0, text="Success!")
                    time.sleep(0.5)
                    my_bar.empty()
                    st.success(f"Successfully ingested {ingested_count} chunks into pgvector!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    # ── Right Column: Search panel ──
    with col_search:
        st.markdown('<h4 style="margin:0 0 0.8rem 0; font-size:0.9rem; color:#f0f0f3;">🔍 Similarity Search Comparison</h4>', unsafe_allow_html=True)
        
        search_query = st.text_input(
            "Search Query",
            placeholder="Type search terms or questions (e.g. 'What is attention mechanism?')",
            key=f"search_query_input_{doc_id}"
        )
        
        search_col1, search_col2 = st.columns([2, 1])
        with search_col1:
            search_mode = st.radio(
                "Retrieval Mode",
                options=["Hybrid (Vector + BM25)", "Vector Search Only (Dense)", "BM25 Search Only (Keyword)"],
                index=0,
                horizontal=True,
                key=f"search_mode_radio_{doc_id}"
            )
        with search_col2:
            search_limit = st.number_input(
                "Limit Results",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                key=f"search_limit_input_{doc_id}"
            )

        col_toggles = st.columns(2)
        with col_toggles[0]:
            show_contextualized = st.toggle(
                "Show enriched contextualized text",
                value=True,
                key=f"search_show_ctx_{doc_id}"
            )
        with col_toggles[1]:
            use_rerank = False
            if "Hybrid" in search_mode:
                use_rerank = st.toggle(
                    "Rerank results with BGE-Reranker-v2-m3",
                    value=False,
                    key=f"search_use_rerank_{doc_id}"
                )
        
        if search_query:
            results = []
            
            with st.spinner("Searching..."):
                try:
                    # 1. Run search based on mode
                    if "Vector Search Only" in search_mode:
                        embedder = get_embedder()
                        query_emb = embedder.get_embeddings([search_query])[0]
                        results = db.vector_search(query_emb, limit=search_limit)
                        mode_label = "vector"
                    elif "BM25 Search Only" in search_mode:
                        results = db.keyword_search(search_query, limit=search_limit)
                        mode_label = "bm25"
                    else:  # Hybrid
                        embedder = get_embedder()
                        query_emb = embedder.get_embeddings([search_query])[0]
                        results = db.hybrid_search(
                            search_query,
                            query_emb,
                            limit=search_limit,
                            use_rerank=use_rerank
                        )
                        mode_label = "hybrid_rerank" if use_rerank else "hybrid"
                        
                    # 2. Render Results
                    if not results:
                        st.markdown(
                            """
                            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:2rem; text-align:center;">
                                <p style="margin:0; font-size:0.82rem; color:#5c5c6e;">No matches found.</p>
                                <p style="margin:4px 0 0 0; font-size:0.72rem; color:#3d3d4a;">Try adjusting your search query or verify chunks have been ingested.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="font-size:0.75rem; color:#5c5c6e; margin-bottom:8px;">Retrieved {len(results)} matches:</div>',
                            unsafe_allow_html=True
                        )
                        
                        for idx, item in enumerate(results):
                            _render_search_result_card(item, idx + 1, mode_label, show_contextualized)
                            
                except Exception as e:
                    st.error(f"Search failed: {e}")


def _render_search_result_card(item: dict, rank: int, mode: str, show_contextualized: bool):
    """Render a single search result as a styled card."""
    # Build tags/metadata
    headings_html = ""
    if item.get("headings"):
        badges = "".join(
            '<span class="chunk-heading-badge">' + html_module.escape(str(h)) + '</span>'
            for h in item["headings"][:3]
        )
        headings_html = '<div class="chunk-headings">' + badges + '</div>'
        
    caption_html = ""
    if item.get("captions"):
        cap_badges = "".join(
            '<span class="chunk-caption-badge">' + html_module.escape(str(c)[:80]) + '</span>'
            for c in item["captions"][:2]
        )
        caption_html = '<div class="chunk-captions">' + cap_badges + '</div>'
        
    type_html = ""
    if item.get("chunk_type"):
        t = item["chunk_type"]
        type_label = t.split(".")[-1] if "." in t else t
        type_html = f'<span class="chunk-type-badge" style="border-color:rgba(96,165,250,0.2);color:#60a5fa;">{html_module.escape(type_label)}</span>'
        
    # Prepare text and truncation
    display_text = item["contextualized"] if show_contextualized else item["text"]
    max_display = 800
    is_truncated = len(display_text) > max_display
    
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

    # Mode-specific score badge
    score_html = ""
    if mode == "vector":
        score_html = f'<div class="search-score" style="color:#34d399;">Cosine Similarity: <strong>{item["score"]:.3f}</strong></div>'
    elif mode == "bm25":
        score_html = f'<div class="search-score" style="color:#fbbf24;">BM25 Score: <strong>{item["score"]:.3f}</strong></div>'
    elif mode == "hybrid_rerank":
        v_score_str = f" ({item['vector_score']:.3f})" if item['vector_score'] is not None else ""
        v_rank = f"#{item['vector_rank']}{v_score_str}" if item['vector_rank'] is not None else "N/A"
        
        f_score_str = f" ({item['fts_score']:.3f})" if item['fts_score'] is not None else ""
        f_rank = f"#{item['fts_rank']}{f_score_str}" if item['fts_rank'] is not None else "N/A"
        
        score_html = (
            f'<div class="search-score" style="color:#a78bfa;">Rerank Score: <strong>{item["rerank_score"]:.4f}</strong>'
            f' <span style="color:#5c5c6e; font-size:0.68rem; margin-left:8px;">(RRF: {item["rrf_score"]:.4f} | Vector Rank: {v_rank} | BM25 Rank: {f_rank})</span>'
            f'</div>'
        )
    else: # hybrid
        v_score_str = f" ({item['vector_score']:.3f})" if item['vector_score'] is not None else ""
        v_rank = f"#{item['vector_rank']}{v_score_str}" if item['vector_rank'] is not None else "N/A"
        
        f_score_str = f" ({item['fts_score']:.3f})" if item['fts_score'] is not None else ""
        f_rank = f"#{item['fts_rank']}{f_score_str}" if item['fts_rank'] is not None else "N/A"
        
        score_html = (
            f'<div class="search-score" style="color:#60a5fa;">RRF Score: <strong>{item["rrf_score"]:.4f}</strong>'
            f' <span style="color:#5c5c6e; font-size:0.68rem; margin-left:8px;">(Vector Rank: {v_rank} | BM25 Rank: {f_rank})</span>'
            f'</div>'
        )

    # Document & page source
    doc_source_html = (
        f'<div style="font-size:0.7rem; color:#5c5c6e; margin-bottom:6px; display:flex; justify-content:space-between;">'
        f'  <span>📄 {html_module.escape(item["doc_name"])}</span>'
        f'  <span style="color:#34d399; font-family:\'JetBrains Mono\',monospace;">Page {item["page_no"]}</span>'
        f'</div>'
    )

    card_style = ""
    if mode == "hybrid_rerank":
        card_style = ' style="border-left-color: #a78bfa;"'

    card_html = (
        f'<div class="search-result-card"{card_style}>'
        '  <div class="search-card-header">'
        '    <div class="search-card-rank">#' + str(rank) + '</div>'
        + score_html
        + type_html
        + '  </div>'
        + doc_source_html
        + headings_html
        + caption_html
        + '<div class="chunk-card-text">' + text_content_html + '</div>'
        '</div>'
    )
    
    st.markdown(card_html, unsafe_allow_html=True)

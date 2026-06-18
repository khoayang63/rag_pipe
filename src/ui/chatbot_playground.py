"""
Chatbot Playground UI component.

Allows users to chat with the ingested documents in Vietnamese using local Ollama models.
"""

import html as html_module
import streamlit as st
import urllib.request

from pipeline.indexer.postgres_store import VectorStore
from pipeline.processing.embedder import get_embedder
from pipeline.processing.chatbot import OllamaChatbot


def render_chatbot_playground(doc_id: str = "0"):
    """Render the conversational Q&A Chatbot Playground tab."""
    st.markdown(
        """
        <div style="margin-bottom:1rem;">
            <h3 style="margin:0; font-size:1.1rem; color:#f0f0f3;">
                💬 Chatbot Playground (Hỏi - Đáp với Tài liệu)
            </h3>
            <p style="margin:4px 0 0 0; font-size:0.78rem; color:#5c5c6e;">
                Đặt câu hỏi bằng tiếng Việt dựa trên nội dung tài liệu đã nạp. 
                Hệ thống truy xuất ngữ cảnh từ pgvector và sinh câu trả lời bằng mô hình <strong>Ollama</strong> chạy cục bộ.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Check Ollama service status ──
    is_ollama_running = True
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    except Exception:
        is_ollama_running = False

    if not is_ollama_running:
        st.markdown(
            """
            <div style="background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.15); 
                        border-radius:8px; padding:14px 16px; margin-bottom:1.5rem;">
                <h4 style="margin:0 0 6px 0; font-size:0.88rem; color:#f87171;">⚠️ Dịch vụ Ollama chưa chạy!</h4>
                <p style="margin:0; font-size:0.78rem; color:#9898a6; line-height:1.45;">
                    Không thể kết nối đến Ollama trên <code>http://localhost:11434</code>. 
                    Vui lòng mở ứng dụng <strong>Ollama Desktop</strong> hoặc chạy lệnh <code>ollama serve</code> trong Terminal 
                    trước khi tiến hành hỏi đáp.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Test DB Connection ──
    db = VectorStore()
    conn_info = db.test_connection()
    if not conn_info["connected"]:
        st.markdown(
            """
            <div style="background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.15); 
                        border-radius:8px; padding:14px 16px; margin-bottom:1.5rem;">
                <h4 style="margin:0 0 6px 0; font-size:0.88rem; color:#f87171;">⚠️ Không kết nối được Database!</h4>
                <p style="margin:0; font-size:0.78rem; color:#9898a6; line-height:1.45;">
                    Ứng dụng RAG yêu cầu cơ sở dữ liệu PostgreSQL hoạt động. Hãy kiểm tra cài đặt Docker hoặc kết nối mạng của bạn.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Configuration Sidebar/Row ──
    chatbot = OllamaChatbot()
    available_models = chatbot.list_local_models()

    col_config1, col_config2, col_config3 = st.columns([2, 1, 1])

    with col_config1:
        # User selects local model
        selected_model = st.selectbox(
            "Mô hình Ollama (LLM Model)",
            options=available_models if available_models else ["qwen2.5:latest"],
            index=0,
            help="Chọn mô hình Ollama đang chạy trên máy của bạn. Khuyên dùng qwen2.5 nhờ hỗ trợ tốt tiếng Việt."
        )

    with col_config2:
        search_limit = st.number_input(
            "Số chunk ngữ cảnh (Top-K)",
            min_value=1,
            max_value=15,
            value=4,
            step=1,
            help="Số lượng phân đoạn tài liệu khớp nhất được gửi làm ngữ cảnh cho LLM trả lời."
        )

    with col_config3:
        use_rerank = st.toggle(
            "Sử dụng Reranker",
            value=False,
            help="Bật mô hình bge-reranker-v2-m3 để chấm điểm lại các chunk và lấy ngữ cảnh chuẩn nhất."
        )

    st.divider()

    # ── Chat history initialization ──
    history_key = f"chat_history_{doc_id}_{selected_model}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    # Clear chat button
    col_clear1, col_clear2 = st.columns([4, 1])
    with col_clear2:
        if st.button("🧹 Xóa hội thoại", use_container_width=True):
            st.session_state[history_key] = []
            st.rerun()

    # Render history
    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Render citations if assistant and citations exist
            if msg["role"] == "assistant" and "citations" in msg and msg["citations"]:
                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                with st.expander("📄 Nguồn tham chiếu (Citations)", expanded=False):
                    for i, item in enumerate(msg["citations"]):
                        idx = item.get("index", "?")
                        page = item.get("page_no", "?")
                        score = item.get("score") or item.get("rrf_score", 0.0)
                        text = item.get("text", "")
                        st.markdown(
                            f"""
                            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); 
                                        border-radius:6px; padding:10px; margin-bottom:8px;">
                                <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#9898a6; margin-bottom:4px;">
                                    <strong>Chunk #{idx} (Trang {page})</strong>
                                    <span>Độ tương đồng: {score:.4f}</span>
                                </div>
                                <div style="font-size:0.78rem; color:#c0c0cc; line-height:1.4;">{html_module.escape(text)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # ── Chat Input & Processing ──
    if user_query := st.chat_input("Hãy hỏi bất kỳ câu hỏi nào về tài liệu..."):
        # 1. Render user query in chat
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Append to state
        st.session_state[history_key].append({"role": "user", "content": user_query})

        # 2. Retrieve Matching Context Chunks
        retrieved_chunks = []
        with st.spinner("Đang truy xuất thông tin từ tài liệu..."):
            try:
                # Generate embedding via BAAI/bge-m3
                embedder = get_embedder()
                query_emb = embedder.get_embeddings([user_query])[0]
                
                # Hybrid search
                retrieved_chunks = db.hybrid_search(
                    user_query,
                    query_emb,
                    limit=search_limit,
                    use_rerank=use_rerank
                )
            except Exception as e:
                st.error(f"Lỗi truy xuất dữ liệu: {e}")

        # 3. Stream Response
        with st.chat_message("assistant"):
            if not retrieved_chunks:
                st.warning("⚠️ Không tìm thấy dữ liệu khớp nào trong Database cho câu hỏi này. Trợ lý sẽ trả lời không kèm ngữ cảnh.")
            
            # Streaming container
            response_placeholder = st.empty()
            full_response = ""

            # Call chatbot generator
            response_generator = chatbot.stream_chat(
                query=user_query,
                chunks=retrieved_chunks,
                model=selected_model,
                history=st.session_state[history_key][:-1]
            )

            for token in response_generator:
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)

            # Show citations expander if chunks exist
            if retrieved_chunks:
                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                with st.expander("📄 Nguồn tham chiếu (Citations)", expanded=False):
                    for i, item in enumerate(retrieved_chunks):
                        idx = item.get("index", "?")
                        page = item.get("page_no", "?")
                        score = item.get("score") or item.get("rrf_score", 0.0)
                        text = item.get("text", "")
                        st.markdown(
                            f"""
                            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); 
                                        border-radius:6px; padding:10px; margin-bottom:8px;">
                                <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#9898a6; margin-bottom:4px;">
                                    <strong>Chunk #{idx} (Trang {page})</strong>
                                    <span>Độ tương đồng: {score:.4f}</span>
                                </div>
                                <div style="font-size:0.78rem; color:#c0c0cc; line-height:1.4;">{html_module.escape(text)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # Append to history state
            st.session_state[history_key].append({
                "role": "assistant",
                "content": full_response,
                "citations": retrieved_chunks
            })
            st.rerun()

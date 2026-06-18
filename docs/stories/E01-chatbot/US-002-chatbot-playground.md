# US-002 Chatbot Playground with Ollama

## Status

in_progress

## Lane

normal

## Product Contract

Enable users to chat with the retrieved chunks of converted documents using Ollama LLM models (e.g. Qwen2.5, Llama3) with citations and session-based conversational memory.

## Relevant Product Docs

- [README.md](file:///c:/Users/OS/Desktop/rag_pipeline/README.md)
- [ARCHITECTURE.md](file:///c:/Users/OS/Desktop/rag_pipeline/ARCHITECTURE.md)

## Acceptance Criteria

- Add a **Chatbot Playground** tab in Streamlit.
- Support selecting Ollama models installed locally (defaults to `qwen2.5` as recommended).
- Allow users to ask questions in Vietnamese.
- Retrieve top matching chunks from `pgvector` database and rerank them using `bge-reranker-v2-m3` if active.
- Formulate a prompt with the context and system instructions.
- Stream the LLM response in real-time.
- Render clickable citations below each assistant answer, linking back to the source chunks.
- Support "Clear Chat" to reset memory.

## Design Notes

- **Backend**: `src/pipeline/processing/chatbot.py` contains the `OllamaChatbot` class communicating with `http://localhost:11434/api/generate`.
- **UI**: `src/ui/chatbot_playground.py` renders the chat interface and model options.
- **Verification**: `tests/test_chatbot.py` verifies query processing and mock LLM calls.

## Validation

| Layer | Expected proof |
| --- | --- |
| Unit | `pytest tests/test_chatbot.py` |
| Integration | Manual chat test with local Ollama service |
| E2E | Run Streamlit app and perform a conversation |

## Harness Delta

- Added `pytest` to unit tests.
- Configured mechanical proof verification command in Harness DB.

## Evidence

- Story registered in Harness DB with ID `US-002`.
- Verification command configured.

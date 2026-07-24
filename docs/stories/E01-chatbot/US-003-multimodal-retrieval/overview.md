# Overview - US-003 Multimodal Image Retrieval

## Current Behavior

The Chatbot Playground only retrieves text document chunks from the PostgreSQL vector database and generates text-only answers. Figures/diagrams are extracted during the document ingestion phase but are only visible in the "Extracted Figures" tab, completely separate from the chat interface.

## Target Behavior

When asking questions in the Chatbot Playground, the system retrieves both relevant text chunks and relevant figures/images (using a visual-linguistic joint embedding space via BGE-VL-large). Under the citations collapsible section, the system displays the retrieved images inline with their captions and VLM-generated visual descriptions.

## Affected Users

- Chatbot users wanting to consult visual diagrams and charts from their documents.
- Systems requiring multimodal evidence retrieval.

## Affected Product Docs

- `docs/product/README.md`
- `docs/ARCHITECTURE.md`

## Non-Goals

- Answering queries by generating new images.
- Performing image editing or generation via LLM.
- Running multimodal query processing over external remote APIs (e.g. GPT-4V, Gemini Pro Vision) — this implementation is fully local offline.

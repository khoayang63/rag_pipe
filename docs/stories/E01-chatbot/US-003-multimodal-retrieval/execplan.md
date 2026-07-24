# Exec Plan - US-003 Multimodal Image Retrieval

## Goal

Enable high-fidelity cross-modal semantic retrieval of images/diagrams from documents based on user text queries, displaying them seamlessly in the Chatbot Playground.

## Scope

In scope:
- Loading model directly using `transformers` with `trust_remote_code=True`.
- Implementing visual/textual embedding with `BAAI/BGE-VL-large`.
- Designing `document_images` DB schema and HNSW indexes.
- Running parallel text and image vector searches.
- Visualizing retrieved images inside citation expanders showing captions and VLM descriptions.

Out of scope:
- Building an image generation or editing pipeline.
- Streaming video files or multi-page PDF rendering in chat.

## Risk Classification

Risk flags:
- **Data model**: Creating new tables and indexes in PostgreSQL.
- **External systems**: Loading HuggingFace models, requiring PyTorch.

Hard gates:
- Loading model on CUDA GPU.
- Running migration logic without losing existing document records.

## Work Phases

1. **Discovery**: Test loading model directly via `transformers` AutoModel.
2. **Design**: Build visual embedder module.
3. **Database Migration**: Add `document_images` table in PostgreSQL.
4. **Wiring Ingestion**: Hook image vector extraction after Docling conversion.
5. **Chatbot Integration**: Update playground UI to search and show images.
6. **Validation**: Test visual queries manually and run automated tests.

## Stop Conditions

Pause for human confirmation if:
- PyTorch/CUDA encounters OOM (Out Of Memory) issues on the target GPU.
- Ingestion fails to process images due to encoding compatibility issues.

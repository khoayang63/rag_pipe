# Validation - US-003 Multimodal Image Retrieval

## Proof Strategy

The story is considered complete once:
1. `tests/test_visual_embedder.py` passes successfully, validating mock model loading and queries.
2. The user can ingest a PDF with figures, ask a conceptual visual question in the Chatbot Playground, and see the correct figure displayed in citations.

## Test Plan

| Layer | Cases |
| --- | --- |
| Unit | Mock visual embedder `encode()` and DB schema creation statements. |
| Integration | Ensure image insertion and retrieval queries execute correctly on pgvector. |
| E2E | Manually upload PDF, ask question in Streamlit UI, view citation card with figure image, caption, and VLM description. |

## Fixtures

- Mock document results and mock figure data.
- Sample test PDF containing a system architecture diagram.

## Commands

```powershell
python -m pytest tests/test_visual_embedder.py
```

## Acceptance Evidence

TBD (to be updated after implementation)

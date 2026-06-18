# Test Matrix

This file maps product behavior to proof.

No product behavior has been defined or implemented yet. Do not mark a row
implemented until tests or validation evidence exist.

## Status Values

| Status | Meaning |
| --- | --- |
| planned | Accepted as intended behavior, not implemented |
| in_progress | Actively being built |
| implemented | Implemented and proof exists |
| changed | Contract changed after earlier implementation |
| retired | No longer part of the product contract |

## Matrix

| Story | Contract | Unit | Integration | E2E | Platform | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [US-002](file:///c:/Users/OS/Desktop/rag_pipeline/docs/stories/E01-chatbot/US-002-chatbot-playground.md) | Chatbot Playground với mô hình Ollama cục bộ (trích dẫn & bộ nhớ hội thoại) | yes | no | no | no | implemented | [test_chatbot.py](file:///c:/Users/OS/Desktop/rag_pipeline/tests/test_chatbot.py) |
| US-TEST | Kiểm tra lỗi cú pháp Python tệp spell_correction_viewer.py | yes | no | no | no | implemented | `python -m py_compile src/ui/spell_correction_viewer.py` |

## Evidence Rules

- Unit proof covers pure domain and application rules.
- Integration proof covers backend enforcement, data integrity, provider
  behavior, jobs, or service contracts.
- E2E proof covers user-visible browser flows.
- Platform proof covers only shell, deployment, mobile, desktop, or runtime
  behavior that cannot be proven in lower layers.
- A story can be implemented without every proof column if the story packet
  explains why.

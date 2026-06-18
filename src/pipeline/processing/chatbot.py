"""
Ollama Chatbot Engine.

Interfaces with local Ollama service to run Vietnamese Q&A over retrieved document chunks.
Uses urllib to avoid external python library dependencies.
"""

import json
import urllib.request
import urllib.error
from typing import Generator, List, Dict, Any


class OllamaChatbot:
    """Helper class to call local Ollama models and generate RAG responses."""

    def __init__(self, host: str = "http://localhost:11434", default_model: str = "qwen2.5:latest"):
        self.host = host.rstrip("/")
        self.default_model = default_model

    def list_local_models(self) -> List[str]:
        """Fetch available models from local Ollama service."""
        try:
            url = f"{self.host}/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                return models
        except Exception as e:
            print(f"[OllamaChatbot] Error listing models: {e}")
            # Return list of standard models as fallbacks if service not yet online
            return ["qwen2.5:latest", "llama3.1:latest", "llama3:latest", "llama3:latest"]

    def generate_prompt(self, query: str, chunks: List[Dict[str, Any]], history: List[Dict[str, str]] = None) -> str:
        """Construct prompt with system instructions, context chunks, and optional chat history."""
        context_parts = []
        for c in chunks:
            idx = c.get("index", "?")
            text = c.get("text", "")
            context_parts.append(f"[Chunk #{idx}]\n{text}")

        context_str = "\n\n".join(context_parts)

        prompt = (
            "Bạn là một trợ lý AI hữu ích, lịch sự và trung thực. Dưới đây là các đoạn thông tin được trích xuất từ tài liệu (Ngữ cảnh):\n"
            "--------------------------------------------------\n"
            f"{context_str}\n"
            "--------------------------------------------------\n\n"
            "Dựa vào ngữ cảnh trên, hãy trả lời câu hỏi của người dùng bằng tiếng Việt:\n"
            "- Hãy trả lời chính xác, trung thực và bám sát vào ngữ cảnh được cung cấp.\n"
            "- Nếu câu trả lời không có trong ngữ cảnh hoặc ngữ cảnh không đủ thông tin, hãy nói rõ \"Tôi không tìm thấy thông tin này trong tài liệu\". Không tự ý bịa đặt câu trả lời.\n"
            "- Ở cuối mỗi ý hoặc cuối câu trả lời, hãy chỉ rõ số thứ tự của đoạn trích dẫn mà bạn tham chiếu từ ngữ cảnh dưới dạng [Chunk #index] (ví dụ: [Chunk #1], [Chunk #3]).\n\n"
        )

        if history:
            prompt += "Lịch sử cuộc trò chuyện:\n"
            for msg in history:
                role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "\n"

        prompt += f"Câu hỏi mới: {query}\nTrả lời:\n"
        return prompt

    def stream_chat(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = None,
        history: List[Dict[str, str]] = None
    ) -> Generator[str, None, None]:
        """Stream generation tokens from Ollama endpoint."""
        selected_model = model or self.default_model
        prompt = self.generate_prompt(query, chunks, history=history)

        url = f"{self.host}/api/generate"
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                for line in response:
                    if not line:
                        continue
                    line_data = json.loads(line.decode("utf-8"))
                    response_text = line_data.get("response", "")
                    if response_text:
                        yield response_text
        except urllib.error.URLError as e:
            yield f"\n\n[ERROR] Không thể kết nối với dịch vụ Ollama cục bộ tại {self.host}. Chi tiết: {e.reason}\n"
            yield "Hãy đảm bảo rằng ứng dụng Ollama đang chạy trên máy tính của bạn."
        except Exception as e:
            yield f"\n\n[ERROR] Đã xảy ra lỗi bất ngờ khi gọi Ollama: {str(e)}"

"""
Unit tests for the Ollama Chatbot Engine.

Tests model listing, prompt generation, and mock stream response generation.
"""

import sys
import os
from unittest.mock import patch, MagicMock

# Add src to sys.path for direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pipeline.processing.chatbot import OllamaChatbot


def test_generate_prompt():
    """Verify prompt formatting with context chunks and user query."""
    chatbot = OllamaChatbot()
    query = "Học máy là gì?"
    chunks = [
        {"index": 1, "text": "Học máy là một lĩnh vực của trí tuệ nhân tạo."},
        {"index": 2, "text": "Mô hình học máy tự động học từ dữ liệu."}
    ]

    prompt = chatbot.generate_prompt(query, chunks)

    assert "[Chunk #1]" in prompt
    assert "Học máy là một lĩnh vực của trí tuệ nhân tạo." in prompt
    assert "[Chunk #2]" in prompt
    assert "Mô hình học máy tự động học từ dữ liệu." in prompt
    assert "Học máy là gì?" in prompt
    assert "Bạn là một trợ lý AI hữu ích, lịch sự và trung thực." in prompt


@patch("urllib.request.urlopen")
def test_list_local_models(mock_urlopen):
    """Verify that models are retrieved correctly from the Ollama tags endpoint."""
    # Mock tags response
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"models": [{"name": "qwen2.5:latest"}, {"name": "llama3.1:latest"}]}'
    mock_urlopen.return_value.__enter__.return_value = mock_response

    chatbot = OllamaChatbot()
    models = chatbot.list_local_models()

    assert len(models) == 2
    assert "qwen2.5:latest" in models
    assert "llama3.1:latest" in models


@patch("urllib.request.urlopen")
def test_stream_chat(mock_urlopen):
    """Verify that chatbot yields tokens successfully from a simulated JSON response stream."""
    # Mock stream response returning multiple JSON lines
    mock_response = [
        b'{"response": "H\xe1\xbb\x8dc "}\n',
        b'{"response": "m\xc3\xa1y "}\n',
        b'{"response": "l\xc3\xa0 "}\n',
        b'{"response": "g\xc3\xac."}\n'
    ]
    
    mock_enter = MagicMock()
    mock_enter.__iter__.return_value = iter(mock_response)
    mock_urlopen.return_value.__enter__.return_value = mock_enter

    chatbot = OllamaChatbot()
    chunks = [{"index": 1, "text": "Học máy là gì"}]
    
    tokens = list(chatbot.stream_chat("Hỏi", chunks, model="qwen2.5:latest"))

    assert len(tokens) == 4
    assert tokens[0] == "Học "
    assert tokens[1] == "máy "
    assert tokens[2] == "là "
    assert tokens[3] == "gì."


def test_generate_prompt_with_history():
    """Verify prompt formatting when chat history is included."""
    chatbot = OllamaChatbot()
    query = "Nêu ứng dụng của nó?"
    chunks = [{"index": 1, "text": "Học máy có nhiều ứng dụng."}]
    history = [
        {"role": "user", "content": "Học máy là gì?"},
        {"role": "assistant", "content": "Học máy là... [Chunk #1]"}
    ]

    prompt = chatbot.generate_prompt(query, chunks, history=history)

    assert "Lịch sử cuộc trò chuyện:" in prompt
    assert "Người dùng: Học máy là gì?" in prompt
    assert "Trợ lý: Học máy là... [Chunk #1]" in prompt
    assert "Câu hỏi mới: Nêu ứng dụng của nó?" in prompt


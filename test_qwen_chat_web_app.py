import os
import sys


sys.path.append(os.path.join(os.getcwd(), "source"))

from qwen_chat_web_app import build_app, normalize_history_payload


class _StubEngine:
    def __init__(self):
        self.calls = []
        self.cleared = []

    def status(self):
        return {
            "loaded": True,
            "device": "cpu",
            "adapter_loaded": True,
            "sessions": 1,
            "profile": {
                "label": "Test Adapter",
                "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "target_modules": ["q_proj", "k_proj"],
                "benchmark": {"available": False},
            },
        }

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "ok": True,
            "session_id": kwargs["session_id"],
            "response": "stub reply",
            "timing": {"total_ms": 12.0, "tokens_per_sec": 5.0, "output_tokens": 3, "preset_used": "balanced"},
        }

    def clear(self, session_id):
        self.cleared.append(session_id)


def smoke_test_history_payload_normalization():
    cleaned = normalize_history_payload(
        [
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": " explain this "},
            {"role": "assistant", "content": "sure"},
            {"role": "bogus", "content": "drop"},
            {"role": "user", "content": ""},
        ]
    )
    assert cleaned == [
        {"role": "user", "content": "explain this"},
        {"role": "assistant", "content": "sure"},
    ], cleaned


def smoke_test_index_includes_enhanced_chat_controls():
    app = build_app(_StubEngine())
    client = app.test_client()
    response = client.get("/")
    html = response.get_data(as_text=True)
    assert "toastRack" in html, "Expected toast rack for non-transcript notifications"
    assert "followupRow" in html, "Expected follow-up action row in composer"
    assert "summary-strip" in html, "Expected conversation summary strip"


def smoke_test_api_chat_accepts_client_history():
    engine = _StubEngine()
    app = build_app(engine)
    client = app.test_client()
    response = client.post(
        "/api/chat",
        json={
            "session_id": "session-123",
            "message": "new message",
            "history": [
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
            ],
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    assert engine.calls, "Expected engine.chat to be called"
    assert engine.calls[-1]["history_override"] == [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ], engine.calls[-1]

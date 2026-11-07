"""Native local Ollama chat client for GKC-AI.

Connects only to an Ollama server controlled by the GKC-AI operator.
No hosted AI API, API key, or third-party inference account is used.
"""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class LocalOllamaProvider:
    """Dependency-free client for Ollama's local /api/chat endpoint."""

    def __init__(self, host, model, timeout=120):
        host = str(host).strip().rstrip("/")
        model = str(model).strip()
        parsed = urlparse(host)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("GKC_AI_HOST must be a valid http(s) URL.")
        if not (parsed.hostname in {"localhost", "127.0.0.1", "::1"} or
                parsed.hostname.endswith(".local")):
            raise ValueError("GKC_AI_HOST must point to a local Ollama server.")
        if not model:
            raise ValueError("A local model name is required (GKC_AI_MODEL).")
        try:
            timeout = float(timeout)
        except (TypeError, ValueError) as error:
            raise ValueError("GKC_AI_TIMEOUT must be a number.") from error
        if timeout <= 0 or timeout > 900:
            raise ValueError("GKC_AI_TIMEOUT must be between 0 and 900 seconds.")
        self.host = host
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls):
        """Return a local provider only when a model name is configured."""
        model = os.getenv("GKC_AI_MODEL", "").strip()
        if not model:
            return None
        return cls(
            host=os.getenv("GKC_AI_HOST", "http://127.0.0.1:11434"),
            model=model,
            timeout=os.getenv("GKC_AI_TIMEOUT", "120"),
        )

    @property
    def endpoint(self):
        return self.host + "/api/chat"

    def complete(self, messages, max_tokens=512):
        if not isinstance(messages, list) or not messages:
            raise ValueError("At least one chat message is required.")
        normalized = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each chat message must be an object.")
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError("Chat message role must be system, user, or assistant.")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Chat message content must be a non-empty string.")
            normalized.append({"role": role, "content": content})
        try:
            token_limit = max(1, min(int(max_tokens), 4096))
        except (TypeError, ValueError) as error:
            raise ValueError("max_tokens must be an integer.") from error

        payload = {
            "model": self.model,
            "messages": normalized,
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": token_limit},
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            detail = ""
            try:
                data = json.loads(error.read().decode("utf-8", errors="replace"))
                detail = data.get("error", "")
            except (ValueError, AttributeError, TypeError):
                pass
            suffix = (": " + str(detail)[:300]) if detail else ""
            raise RuntimeError(f"Local Ollama returned HTTP {error.code}{suffix}") from error
        except URLError as error:
            raise RuntimeError(f"Cannot reach local Ollama: {error.reason}") from error
        except TimeoutError as error:
            raise RuntimeError("Local model request timed out.") from error
        try:
            content = json.loads(raw)["message"]["content"]
        except (ValueError, KeyError, TypeError) as error:
            raise RuntimeError("Local Ollama returned an unexpected response.") from error
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Local Ollama returned an empty answer.")
        return content.strip()

"""OpenAI-compatible chat-completions client for GKC-AI.

Uses only the Python standard library so the assistant can connect to hosted
providers or local servers exposing /v1/chat/completions (for example Ollama).
Credentials are read from the process environment and are never logged.
"""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class OpenAICompatibleProvider:
    """Minimal, dependency-free OpenAI-compatible chat-completions client."""

    def __init__(self, base_url, model, api_key="", timeout=45):
        base_url = str(base_url).strip().rstrip("/")
        model = str(model).strip()
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("GKC_AI_BASE_URL must be a valid http(s) URL.")
        if not model:
            raise ValueError("A model name is required (GKC_AI_MODEL).")
        try:
            timeout = float(timeout)
        except (TypeError, ValueError) as error:
            raise ValueError("GKC_AI_TIMEOUT must be a number.") from error
        if timeout <= 0 or timeout > 300:
            raise ValueError("GKC_AI_TIMEOUT must be between 0 and 300 seconds.")

        self.base_url = base_url
        self.model = model
        self.api_key = str(api_key or "").strip()
        self.timeout = timeout

    @classmethod
    def from_env(cls):
        """Build a provider from environment variables, or return None if unset."""
        model = os.getenv("GKC_AI_MODEL", "").strip()
        if not model:
            return None
        return cls(
            base_url=os.getenv("GKC_AI_BASE_URL", "https://api.openai.com/v1"),
            model=model,
            api_key=os.getenv("GKC_AI_API_KEY", ""),
            timeout=os.getenv("GKC_AI_TIMEOUT", "45"),
        )

    @property
    def endpoint(self):
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def complete(self, messages, max_tokens=256):
        """Return the assistant text for a list of role/content messages."""
        if not isinstance(messages, list) or not messages:
            raise ValueError("At least one chat message is required.")
        normalized = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each chat message must be an object.")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError("Chat message role must be system, user, or assistant.")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Chat message content must be a non-empty string.")
            normalized.append({"role": role, "content": content})

        try:
            token_limit = max(1, min(int(max_tokens), 2048))
        except (TypeError, ValueError) as error:
            raise ValueError("max_tokens must be an integer.") from error

        payload = {
            "model": self.model,
            "messages": normalized,
            "max_tokens": token_limit,
            "temperature": 0.3,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            detail = ""
            try:
                body = error.read().decode("utf-8", errors="replace")
                parsed = json.loads(body)
                detail = parsed.get("error", {}).get("message", "")
            except (ValueError, AttributeError, TypeError):
                detail = ""
            suffix = (": " + str(detail)[:300]) if detail else ""
            raise RuntimeError(
                f"Language-model API returned HTTP {error.code}{suffix}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"Cannot reach language-model API: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise RuntimeError("Language-model API request timed out.") from error

        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "Language-model API returned an unexpected response format."
            ) from error

        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Language-model API returned an empty answer.")
        return content.strip()

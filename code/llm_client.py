"""Unified LLM client supporting OpenAI and Anthropic."""

import os
import json
from typing import Optional, Dict, Any

from config import LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY


class LLMClient:
    """Simple wrapper around OpenAI and Anthropic APIs with structured output."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or LLM_PROVIDER).lower()
        self.model = model or LLM_MODEL
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        return self._client

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.1, max_tokens: int = 1500) -> str:
        """Send a chat request and return the text response."""
        client = self._get_client()
        if self.provider == "openai":
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        elif self.provider == "anthropic":
            resp = client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.content[0].text if resp.content else ""
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def structured_chat(self, system_prompt: str, user_prompt: str,
                        json_schema: Dict[str, Any],
                        temperature: float = 0.1,
                        max_tokens: int = 1500) -> Dict[str, Any]:
        """Request structured JSON output. Falls back to text parsing if unsupported."""
        client = self._get_client()
        full_prompt = (
            f"{system_prompt}\n\n"
            f"You MUST respond with a single valid JSON object matching this schema:\n"
            f"{json.dumps(json_schema, indent=2)}\n\n"
            f"Do not include markdown formatting or explanations outside the JSON.\n\n"
            f"User query:\n{user_prompt}"
        )

        if self.provider == "openai":
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": full_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content or "{}"
                return json.loads(text)
            except Exception:
                # Fallback
                text = self.chat(system_prompt, user_prompt, temperature, max_tokens)
                return self._extract_json(text)
        else:
            text = self.chat(system_prompt, user_prompt, temperature, max_tokens)
            return self._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON from markdown or raw text."""
        import re
        # Try fenced code block
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        # Try raw object
        m = re.search(r'(\{.*\})', text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        return {}


def get_client() -> LLMClient:
    return LLMClient()


if __name__ == "__main__":
    c = get_client()
    print(c.chat("You are a helper.", "Say hello in JSON format."))

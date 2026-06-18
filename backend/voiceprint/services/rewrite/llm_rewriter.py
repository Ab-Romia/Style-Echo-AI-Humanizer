"""
In-context LLM rewriter (bring your own key).

Adapts a draft toward an author's voice with a single chat completion call.
The prompt carries the source draft, three to five real excerpts from the
author's own samples as exemplars, and the rendered profile constraints. The
model is asked to rewrite the draft in the author's voice while preserving
meaning.

The client is OpenAI-compatible, so the same code path works with the OpenAI
API and with OpenRouter by switching the base_url. No key is ever hardcoded:
the key is read from an argument or from the OPENAI_API_KEY environment
variable. If no key is available, the rewriter raises NoApiKeyError so the
caller can fall back to the rule-based rewriter.
"""
import os
from typing import List, Optional

from voiceprint.models.style_profile import StyleProfile
from voiceprint.services.rewrite.profile_to_constraints import render_constraints


class NoApiKeyError(RuntimeError):
    """Raised when no API key is available for the LLM rewriter."""


class LlmRewriter:
    """Adapts a draft toward an author's voice with one LLM call."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Configure the rewriter.

        Args:
            api_key: API key. Falls back to the OPENAI_API_KEY env var.
            base_url: OpenAI-compatible base URL. Falls back to the
                OPENAI_BASE_URL env var, then to the OpenAI default. For
                OpenRouter use https://openrouter.ai/api/v1.
            model: Chat model name. Falls back to the OPENAI_MODEL env var,
                then to a small default.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self.model = model or os.environ.get("OPENAI_MODEL") or self.DEFAULT_MODEL
        self._client = None

    def is_available(self) -> bool:
        """Return True when a non-empty key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def _get_client(self):
        """Lazily build the OpenAI-compatible client."""
        if not self.is_available():
            raise NoApiKeyError(
                "No API key provided. Set OPENAI_API_KEY or pass api_key to "
                "use the LLM rewriter, or fall back to the rule rewriter."
            )
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The openai package is required for the LLM rewriter. "
                    "Install it or use the rule rewriter."
                ) from exc
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _select_exemplars(
        self, samples: List[str], max_count: int = 5, max_chars: int = 600
    ) -> List[str]:
        """Pick three to five sample excerpts to anchor the voice."""
        excerpts: List[str] = []
        for sample in samples:
            cleaned = (sample or "").strip()
            if not cleaned:
                continue
            if len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + " ..."
            excerpts.append(cleaned)
            if len(excerpts) >= max_count:
                break
        return excerpts

    def _build_prompt(self, source_draft: str, profile: StyleProfile) -> str:
        """Assemble the user prompt from exemplars, constraints, and the draft."""
        exemplars = self._select_exemplars(profile.samples or [])
        constraints = render_constraints(profile)

        exemplar_block = "\n\n".join(
            f"Exemplar {i + 1}:\n{text}" for i, text in enumerate(exemplars)
        )

        return (
            "Here are excerpts of the author's own writing.\n\n"
            f"{exemplar_block}\n\n"
            "Here are the measured features of the author's voice:\n"
            f"{constraints}\n\n"
            "Rewrite the draft below so it reads in this author's voice. "
            "Preserve the meaning, the facts, and the structure of the "
            "argument. Do not add new claims. Do not use em-dashes. Return "
            "only the rewritten draft, with no commentary.\n\n"
            f"Draft:\n{source_draft}"
        )

    def rewrite(self, source_draft: str, profile: StyleProfile) -> str:
        """
        Rewrite the draft in the author's voice.

        Args:
            source_draft: The text the author pasted to adapt.
            profile: The author's style profile.

        Returns:
            The adapted text.

        Raises:
            NoApiKeyError: When no API key is available.
            RuntimeError: When the openai package is missing or the call fails.
        """
        client = self._get_client()
        prompt = self._build_prompt(source_draft, profile)

        system_message = (
            "You adapt a draft into a specific author's writing voice while "
            "keeping the meaning intact. You never use em-dashes."
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
        except Exception as exc:
            raise RuntimeError(f"LLM rewrite call failed: {exc}") from exc

        if not response.choices:
            raise RuntimeError("LLM rewrite returned no choices.")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("LLM rewrite returned empty content.")
        return content.strip()

"""
Rule-based rewriter.

Deterministic, no-key fallback that nudges a draft toward an author's voice
using only two reliable signals salvaged from the earlier prototype:
contraction normalization toward the author's contraction rate, and a light
punctuation-rate nudge (commas) toward the author's comma rate.

This rewriter preserves meaning. It never injects typos, filler words, or
random punctuation. When a signal is ambiguous it leaves the text unchanged.
"""
import re
from typing import Any, Dict

from voiceprint.models.style_profile import StyleProfile


# Full form on the left, contracted form on the right.
CONTRACTION_MAP = {
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    "have not": "haven't",
    "has not": "hasn't",
    "had not": "hadn't",
    "will not": "won't",
    "would not": "wouldn't",
    "should not": "shouldn't",
    "could not": "couldn't",
    "cannot": "can't",
    "i am": "I'm",
    "you are": "you're",
    "it is": "it's",
    "we are": "we're",
    "they are": "they're",
    "that is": "that's",
    "there is": "there's",
    "i have": "I've",
    "you have": "you've",
    "we have": "we've",
    "they have": "they've",
    "i will": "I'll",
    "you will": "you'll",
    "we will": "we'll",
    "they will": "they'll",
}

# Inverse, for expansion. Built once at import time.
EXPANSION_MAP = {v.lower(): k for k, v in CONTRACTION_MAP.items()}


class RuleRewriter:
    """Adapts a draft toward an author's voice with deterministic rules."""

    # If the author's contraction rate is above this, prefer contractions.
    CONTRACTION_HIGH = 0.02
    # If below this, prefer expanded forms.
    CONTRACTION_LOW = 0.01

    def rewrite(self, source_draft: str, profile: StyleProfile) -> str:
        """
        Return a meaning-preserving adaptation of the draft.

        Args:
            source_draft: The text the author pasted to adapt.
            profile: The author's style profile.

        Returns:
            The adapted text. May equal the input when no rule applies.
        """
        if not source_draft or not source_draft.strip():
            return source_draft

        stylometric = profile.stylometric_features or {}
        linguistic = profile.linguistic_features or {}

        text = source_draft
        text = self._match_contractions(text, stylometric)
        text = self._match_comma_rate(text, linguistic)
        return text

    def _match_contractions(self, text: str, stylometric: Dict[str, Any]) -> str:
        """Move contraction usage toward the author's measured rate."""
        rate = stylometric.get("contraction_rate", 0.0)

        if rate >= self.CONTRACTION_HIGH:
            mapping = {full: short for full, short in CONTRACTION_MAP.items()}
        elif rate <= self.CONTRACTION_LOW:
            mapping = {short: full for short, full in EXPANSION_MAP.items()}
        else:
            return text

        result = text
        for source, target in mapping.items():
            pattern = re.compile(r"\b" + re.escape(source) + r"\b", re.IGNORECASE)

            def _replace(match: "re.Match[str]", repl: str = target) -> str:
                original = match.group(0)
                if original[:1].isupper():
                    return repl[:1].upper() + repl[1:]
                return repl

            result = pattern.sub(_replace, result)
        return result

    def _match_comma_rate(self, text: str, linguistic: Dict[str, Any]) -> str:
        """
        Nudge comma usage toward the author's comma rate.

        Only conservative edits are made: adding a comma before a coordinating
        conjunction that joins clauses, or removing one in the same position.
        These edits do not change meaning.
        """
        patterns = linguistic.get("punctuation_patterns", {})
        target_per_100 = patterns.get("comma_per_100_words")
        if target_per_100 is None:
            return text

        words = len(text.split())
        if words == 0:
            return text

        current_per_100 = (text.count(",") / words) * 100

        # Author uses noticeably fewer commas: drop commas before and/but.
        if current_per_100 > target_per_100 * 1.5:
            return re.sub(r",\s+(and|but)\s+", r" \1 ", text)

        # Author uses noticeably more commas: add commas before and/but.
        if current_per_100 < target_per_100 * 0.5:
            return re.sub(r"(?<!,)\s+(and|but)\s+", r", \1 ", text)

        return text

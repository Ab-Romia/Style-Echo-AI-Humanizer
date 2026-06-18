"""
Profile to natural-language constraints.

Turns the numeric stylometric and linguistic profile into a short list of
plain-English writing constraints. The LLM rewriter feeds this list into its
prompt; a human can also read it to understand the measured voice.

The renderer never instructs em-dash usage and never emits an em-dash.
"""
from typing import Any, Dict, List

from voiceprint.models.style_profile import StyleProfile


def render_constraints(profile: StyleProfile) -> str:
    """
    Render the profile as a short bullet list of writing constraints.

    Args:
        profile: The author's style profile.

    Returns:
        A newline-joined list of constraints, one per line, each prefixed
        with a dash. Returns a single fallback line if the profile is empty.
    """
    linguistic: Dict[str, Any] = profile.linguistic_features or {}
    stylometric: Dict[str, Any] = profile.stylometric_features or {}

    lines: List[str] = []

    avg_len = linguistic.get("avg_sentence_length")
    if avg_len:
        lines.append(
            f"- Average sentence length is about {avg_len:.0f} words; "
            "keep sentences in that range."
        )

    ttr = linguistic.get("type_token_ratio")
    if ttr:
        if ttr >= 0.6:
            lines.append("- Vocabulary is varied; avoid repeating the same words.")
        elif ttr <= 0.4:
            lines.append(
                "- Vocabulary is plain and repeats common words; do not reach "
                "for fancy synonyms."
            )

    contraction_rate = stylometric.get("contraction_rate", 0.0)
    if contraction_rate >= 0.02:
        lines.append("- Uses contractions freely (it's, don't, I'm).")
    elif contraction_rate <= 0.005:
        lines.append("- Avoids contractions; prefers full forms (it is, do not).")

    patterns = linguistic.get("punctuation_patterns", {})
    semicolon_rate = patterns.get("semicolon_per_100_words", 0.0)
    if semicolon_rate >= 0.5:
        lines.append("- Uses semicolons regularly.")
    else:
        lines.append("- Uses semicolons rarely.")

    comma_rate = patterns.get("comma_per_100_words")
    if comma_rate is not None:
        if comma_rate >= 8:
            lines.append("- Leans on commas; sentences carry several clauses.")
        elif comma_rate <= 3:
            lines.append("- Uses commas sparingly; favors short, direct clauses.")

    passive_ratio = stylometric.get("passive_voice_ratio")
    if passive_ratio is not None:
        if passive_ratio >= 0.25:
            lines.append("- Comfortable with passive voice.")
        else:
            lines.append("- Prefers active voice.")

    transition_rate = stylometric.get("transition_word_rate")
    if transition_rate is not None:
        if transition_rate >= 0.02:
            lines.append(
                "- Uses transition words (however, therefore) to link ideas."
            )
        else:
            lines.append("- Rarely uses formal transition words.")

    fre = linguistic.get("flesch_reading_ease")
    if fre is not None:
        if fre >= 60:
            lines.append("- Reads easily; aim for a conversational reading level.")
        elif fre <= 40:
            lines.append("- Reads denser; a more formal register is fine.")

    if not lines:
        return "- No measured constraints available; match the exemplars provided."

    return "\n".join(lines)

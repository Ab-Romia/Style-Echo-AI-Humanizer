"""
Feedback builder.

Produces the report a user sees after adapting a draft:
- radar axes (normalized stylometric features) for the draft before and after,
  ready for a UI to draw two overlaid polygons against the author's profile;
- the per-axis delta between before and after;
- a per-sentence diff labeling each sentence improved, regressed, or same,
  based on whether its voice match to the author centroid moved.

Framing stays honest: this is voice-guided adaptation, not impersonation.
"""
from typing import Any, Dict, List

import numpy as np


# Each axis maps a raw feature to a [0, 1] radar value by dividing by a
# reference scale and clamping. The scales are rough upper bounds for typical
# English prose, chosen so common values land in the middle of the axis. They
# are display aids, not statistical claims.
RADAR_AXES = {
    "avg_sentence_length": ("linguistic", 40.0),
    "type_token_ratio": ("linguistic", 1.0),
    "contraction_rate": ("stylometric", 0.1),
    "passive_voice_ratio": ("stylometric", 1.0),
    "function_word_ratio": ("stylometric", 0.6),
    "transition_word_rate": ("stylometric", 0.1),
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_axes(
    linguistic: Dict[str, Any], stylometric: Dict[str, Any]
) -> Dict[str, float]:
    """
    Map raw features onto normalized radar axes in [0, 1].

    Args:
        linguistic: Linguistic features for one text.
        stylometric: Stylometric features for one text.

    Returns:
        Axis name to normalized value.
    """
    sources = {"linguistic": linguistic, "stylometric": stylometric}
    axes: Dict[str, float] = {}
    for axis, (family, scale) in RADAR_AXES.items():
        raw = sources[family].get(axis, 0.0) or 0.0
        axes[axis] = _clamp01(raw / scale) if scale else 0.0
    return axes


def build_feedback(
    source_draft: str,
    adapted_text: str,
    profile,
    style_embedder,
    linguistic_analyzer,
    stylometric_analyzer,
    nlp=None,
) -> Dict[str, Any]:
    """
    Build the radar comparison and per-sentence diff.

    Args:
        source_draft: The text the author pasted to adapt.
        adapted_text: The adapted output.
        profile: The author's style profile (for the reference polygon).
        style_embedder: A StyleEmbedder, used for per-sentence voice match.
        linguistic_analyzer: A LinguisticAnalyzer.
        stylometric_analyzer: A StylometricAnalyzer.
        nlp: Optional spaCy pipeline for sentence splitting.

    Returns:
        Dict with keys: radar (author/before/after axes), axis_delta,
        sentence_diff.
    """
    before_axes = normalize_axes(
        linguistic_analyzer.analyze_text(source_draft),
        stylometric_analyzer.analyze_text(source_draft),
    )
    after_axes = normalize_axes(
        linguistic_analyzer.analyze_text(adapted_text),
        stylometric_analyzer.analyze_text(adapted_text),
    )
    author_axes = normalize_axes(
        profile.linguistic_features or {},
        profile.stylometric_features or {},
    )

    axis_delta = {
        axis: round(after_axes[axis] - before_axes[axis], 4)
        for axis in RADAR_AXES
    }

    sentence_diff = _build_sentence_diff(
        source_draft, adapted_text, profile, style_embedder, nlp
    )

    return {
        "radar": {
            "axes": list(RADAR_AXES.keys()),
            "author": author_axes,
            "before": before_axes,
            "after": after_axes,
        },
        "axis_delta": axis_delta,
        "sentence_diff": sentence_diff,
    }


def _build_sentence_diff(
    source_draft: str,
    adapted_text: str,
    profile,
    style_embedder,
    nlp=None,
    threshold: float = 0.02,
) -> List[Dict[str, Any]]:
    """
    Label each adapted sentence improved, regressed, or same.

    The label compares the per-sentence voice match of the adapted text against
    the average per-sentence voice match of the source draft. The threshold is
    a small dead zone so tiny numeric wobble does not flip the label.
    """
    centroid = profile.get_style_centroid_array()
    if centroid is None or np.size(centroid) == 0:
        # Without a style centroid we cannot score sentences honestly.
        return []

    before_pairs = style_embedder.per_sentence_match(source_draft, centroid, nlp)
    after_pairs = style_embedder.per_sentence_match(adapted_text, centroid, nlp)

    if before_pairs:
        baseline = float(np.mean([score for _, score in before_pairs]))
    else:
        baseline = 0.0

    diff: List[Dict[str, Any]] = []
    for sentence, score in after_pairs:
        if score > baseline + threshold:
            label = "improved"
        elif score < baseline - threshold:
            label = "regressed"
        else:
            label = "same"
        diff.append(
            {
                "sentence": sentence,
                "voice_match": round(score, 4),
                "label": label,
            }
        )
    return diff

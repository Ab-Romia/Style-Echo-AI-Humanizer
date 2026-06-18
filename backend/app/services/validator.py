"""
Validation and Similarity Scoring System.

Validates the quality of an adapted draft by checking:
- Cosine similarity with the style centroid
- Linguistic feature alignment
- Readability score match
- Semantic meaning preservation against the author's source draft
"""
from typing import Dict, Any, List
import numpy as np
from app.services.linguistic_analyzer import LinguisticAnalyzer
from app.services.embedding_analyzer import EmbeddingAnalyzer
from app.models.style_profile import StyleProfile


class OutputValidator:
    """Validates the quality of an adapted draft."""

    def __init__(
        self,
        linguistic_analyzer: LinguisticAnalyzer,
        embedding_analyzer: EmbeddingAnalyzer,
    ):
        """
        Initialize the validator.

        Args:
            linguistic_analyzer: LinguisticAnalyzer instance
            embedding_analyzer: EmbeddingAnalyzer instance
        """
        self.linguistic_analyzer = linguistic_analyzer
        self.embedding_analyzer = embedding_analyzer

    def validate_output(
        self,
        source_draft: str,
        adapted_text: str,
        profile: StyleProfile,
    ) -> Dict[str, Any]:
        """
        Validate the adapted draft against the style profile.

        Args:
            source_draft: The text the author pasted to adapt
            adapted_text: The adapted output text
            profile: The author's style profile

        Returns:
            Dictionary with validation metrics
        """
        # 1. Compute style similarity
        style_similarity = self._compute_style_similarity(adapted_text, profile)

        # 2. Check linguistic feature alignment
        feature_match = self._check_feature_alignment(adapted_text, profile)

        # 3. Check readability match
        readability_match = self._check_readability_match(adapted_text, profile)

        # 4. Verify the adapted text still means the same as the source draft
        semantic_preservation = self._verify_semantic_preservation(
            source_draft, adapted_text
        )

        # 5. Calculate overall quality score
        overall_quality = self._calculate_quality_score(
            style_similarity,
            feature_match,
            readability_match,
            semantic_preservation,
        )

        return {
            "style_similarity": style_similarity,
            "feature_alignment": feature_match,
            "readability_match": readability_match,
            "semantic_preservation": semantic_preservation,
            "overall_quality_score": overall_quality,
            "passes_validation": overall_quality >= 0.75,
        }

    def _compute_style_similarity(
        self, text: str, profile: StyleProfile
    ) -> float:
        """
        Compute cosine similarity with style centroid.

        Args:
            text: Text to evaluate
            profile: Style profile

        Returns:
            Similarity score (0-1)
        """
        if profile.style_centroid is None:
            return 0.0

        similarity = self.embedding_analyzer.compute_centroid_similarity(
            text, profile.style_centroid
        )

        return similarity

    def _check_feature_alignment(
        self, text: str, profile: StyleProfile
    ) -> Dict[str, float]:
        """
        Check how well linguistic features align with profile.

        Args:
            text: Text to evaluate
            profile: Style profile

        Returns:
            Dictionary with feature match scores
        """
        # Analyze text features
        text_features = self.linguistic_analyzer.analyze_text(text)
        profile_features = profile.linguistic_features

        # The 0.75 pass threshold and the scaling constants below (the *5 on
        # type-token-ratio difference and the *10 on punctuation-density
        # difference) are heuristics tuned on small samples to turn a raw
        # feature gap into a [0, 1] match. They are reasonable starting points,
        # not arbitrary, and would be recalibrated on a larger labeled set.
        matches = {}

        # Compare sentence length
        if "avg_sentence_length" in profile_features:
            target_avg = profile_features["avg_sentence_length"]
            target_std = profile_features.get("sentence_length_std", 5)
            actual_avg = text_features.get("avg_sentence_length", 0)

            # Within 1 std dev is good
            diff = abs(actual_avg - target_avg)
            matches["sentence_length_match"] = max(0, 1 - (diff / (target_std * 2)))

        # Compare type-token ratio
        if "type_token_ratio" in profile_features:
            target_ttr = profile_features["type_token_ratio"]
            actual_ttr = text_features.get("type_token_ratio", 0)

            diff = abs(actual_ttr - target_ttr)
            matches["lexical_diversity_match"] = max(0, 1 - (diff * 5))

        # Compare punctuation density
        if "punctuation_density" in profile_features:
            target_pd = profile_features["punctuation_density"]
            actual_pd = text_features.get("punctuation_density", 0)

            diff = abs(actual_pd - target_pd)
            matches["punctuation_match"] = max(0, 1 - (diff * 10))

        return matches

    def _check_readability_match(
        self, text: str, profile: StyleProfile
    ) -> bool:
        """
        Check if readability falls within user's typical range.

        Args:
            text: Text to evaluate
            profile: Style profile

        Returns:
            True if readability matches, False otherwise
        """
        text_features = self.linguistic_analyzer.analyze_text(text)
        profile_features = profile.linguistic_features

        # Check Flesch Reading Ease
        if "flesch_reading_ease" in profile_features:
            target_fre = profile_features["flesch_reading_ease"]
            actual_fre = text_features.get("flesch_reading_ease", 0)

            # Within 20 points is acceptable
            if abs(actual_fre - target_fre) > 20:
                return False

        return True

    def _verify_semantic_preservation(
        self, source_draft: str, adapted_text: str
    ) -> float:
        """
        Verify that meaning is preserved against the author's source draft.

        Args:
            source_draft: The text the author pasted to adapt
            adapted_text: The adapted output

        Returns:
            Semantic similarity score (0-1)
        """
        # Compare the adapted output back against the source draft, so a high
        # score means the adaptation kept the author's intended meaning.
        similarity = self.embedding_analyzer.compute_text_similarity(
            source_draft, adapted_text
        )

        return similarity

    def _calculate_quality_score(
        self,
        style_similarity: float,
        feature_match: Dict[str, float],
        readability_match: bool,
        semantic_preservation: float,
    ) -> float:
        """
        Calculate overall quality score.

        Args:
            style_similarity: Style centroid similarity
            feature_match: Feature alignment scores
            readability_match: Whether readability matches
            semantic_preservation: Semantic preservation score

        Returns:
            Overall quality score (0-1)
        """
        # Weighted average
        weights = {
            "style_similarity": 0.35,
            "feature_match": 0.25,
            "readability": 0.15,
            "semantic": 0.25,
        }

        # Average feature match scores
        avg_feature_match = (
            sum(feature_match.values()) / len(feature_match)
            if feature_match
            else 0.0
        )

        # Calculate weighted score
        score = (
            style_similarity * weights["style_similarity"]
            + avg_feature_match * weights["feature_match"]
            + (1.0 if readability_match else 0.5) * weights["readability"]
            + semantic_preservation * weights["semantic"]
        )

        return score

    def suggest_improvements(
        self, validation_results: Dict[str, Any]
    ) -> List[str]:
        """
        Suggest improvements based on validation results.

        Args:
            validation_results: Results from validate_output

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Check style similarity
        if validation_results["style_similarity"] < 0.8:
            suggestions.append(
                "Voice match to your profile is low; the draft still reads "
                "some distance from your samples."
            )

        # Check feature alignment
        feature_alignment = validation_results["feature_alignment"]

        if "sentence_length_match" in feature_alignment:
            if feature_alignment["sentence_length_match"] < 0.7:
                suggestions.append(
                    "Sentence lengths differ from your usual range."
                )

        if "punctuation_match" in feature_alignment:
            if feature_alignment["punctuation_match"] < 0.7:
                suggestions.append(
                    "Punctuation patterns differ from your usual habits."
                )

        # Check semantic preservation
        if validation_results["semantic_preservation"] < 0.85:
            suggestions.append(
                "The adapted text drifted from your source draft; check that "
                "the meaning still holds."
            )

        # Check overall quality
        if validation_results["overall_quality_score"] < 0.75:
            suggestions.append(
                "Overall quality is below threshold. Consider running another iteration."
            )

        if not suggestions:
            suggestions.append("Output looks good! Quality metrics are within acceptable ranges.")

        return suggestions

"""
Main VoicePrint Service.

Brings together all components to provide the complete text humanization pipeline.
"""
from typing import Dict, Any, List
import logging
from app.models.style_profile import StyleProfile, profile_store
from app.services.linguistic_analyzer import LinguisticAnalyzer
from app.services.stylometric_analyzer import StylometricAnalyzer
from app.services.embedding_analyzer import EmbeddingAnalyzer
from app.services.ai_detector_remover import AIDetectorRemover
from app.services.style_transfer import StyleTransferEngine
from app.services.validator import HumanizationValidator

logger = logging.getLogger(__name__)


class VoicePrintService:
    """Main service that orchestrates the entire humanization pipeline."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Set up all the analysis components."""
        self.linguistic_analyzer = LinguisticAnalyzer(spacy_model)
        self.stylometric_analyzer = StylometricAnalyzer(spacy_model)
        self.embedding_analyzer = EmbeddingAnalyzer()
        self.ai_remover = AIDetectorRemover(spacy_model)
        self.style_transfer = StyleTransferEngine(spacy_model)
        self.validator = HumanizationValidator(
            self.linguistic_analyzer,
            self.embedding_analyzer
        )

    def create_style_profile(
        self,
        user_id: str,
        samples: List[str],
        profile_name: str = None,
    ) -> StyleProfile:
        """
        Build a complete writing style profile from user samples.

        Takes 3-10 text samples and extracts all the linguistic patterns,
        stylometric markers, and creates an embedding centroid.
        """
        logger.info(f"Building style profile for user {user_id}")

        # Basic validation
        total_words = sum(len(sample.split()) for sample in samples)
        if total_words < 500:
            raise ValueError(
                f"Need at least 500 words total. You provided {total_words} words."
            )

        # Build the profile
        profile = StyleProfile(
            user_id=user_id,
            samples=samples,
            profile_name=profile_name
        )

        # Extract linguistic features
        logger.info("Pulling out linguistic patterns...")
        linguistic_features = self.linguistic_analyzer.analyze_multiple_samples(samples)
        profile.set_linguistic_features(linguistic_features)

        # Extract stylometric features
        logger.info("Analyzing writing style markers...")
        stylometric_features = self.stylometric_analyzer.analyze_multiple_samples(samples)
        profile.set_stylometric_features(stylometric_features)

        # Create embedding centroid
        logger.info("Generating style embedding...")
        centroid = self.embedding_analyzer.create_style_centroid(samples)

        # Check consistency
        consistency = self.embedding_analyzer.analyze_style_consistency(samples)

        profile.set_style_centroid(centroid, {
            "embedding_dimension": len(centroid),
            "consistency_metrics": consistency,
        })

        # Save to store
        profile_store.save_profile(profile)

        logger.info(f"Profile created successfully: {profile.profile_id}")
        return profile

    def humanize_text(
        self,
        profile_id: str,
        ai_text: str,
        strength: float = 0.7,
        preserve_meaning: bool = True,
    ) -> Dict[str, Any]:
        """
        Transform AI-generated text to match a user's writing style.

        First removes obvious AI tells, then applies the user's specific
        writing patterns and validates the result.
        """
        logger.info(f"Humanizing text with profile {profile_id}")

        # Get the profile
        profile = profile_store.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")

        # Step 1: Remove AI detection patterns
        logger.info("Removing AI fingerprints...")
        ai_removal_result = self.ai_remover.analyze_and_remove(ai_text, strength)
        deAI_text = ai_removal_result["modified_text"]

        # Step 2: Apply user's writing style
        logger.info("Applying your writing style...")
        humanized_text = self.style_transfer.apply_style(
            deAI_text,
            profile,
            strength
        )

        # Step 3: Validate the output
        logger.info("Checking the results...")
        validation = self.validator.validate_output(
            ai_text,
            humanized_text,
            profile
        )

        # If validation fails and user wants to preserve meaning
        if not validation["passes_validation"] and preserve_meaning:
            if validation["semantic_preservation"] < 0.85:
                logger.warning("Meaning changed too much, trying again with lower strength")
                # Retry with lower strength
                return self.humanize_text(
                    profile_id,
                    ai_text,
                    strength * 0.8,
                    preserve_meaning
                )

        # Get suggestions for improvement
        suggestions = self.validator.suggest_improvements(validation)

        return {
            "original_text": ai_text,
            "humanized_text": humanized_text,
            "ai_removal_metrics": {
                "original_ai_score": ai_removal_result["original_metrics"]["ai_probability"],
                "improved_ai_score": ai_removal_result["modified_metrics"]["ai_probability"],
                "improvement": ai_removal_result["improvement"],
            },
            "validation": validation,
            "suggestions": suggestions,
            "metadata": {
                "profile_id": profile_id,
                "strength": strength,
                "iterations": 1,
            }
        }

    def get_profile(self, profile_id: str) -> Dict[str, Any]:
        """Get profile details by ID."""
        profile = profile_store.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")

        return profile.to_dict()

    def list_user_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all profiles for a specific user."""
        profiles = profile_store.get_user_profiles(user_id)
        return [p.to_dict() for p in profiles]

    def delete_profile(self, profile_id: str) -> bool:
        """Remove a profile permanently."""
        return profile_store.delete_profile(profile_id)

    def quick_analysis(self, text: str) -> Dict[str, Any]:
        """
        Quick analysis of any text without needing a profile.
        Useful for testing or previewing what we extract.
        """
        linguistic = self.linguistic_analyzer.analyze_text(text)
        stylometric = self.stylometric_analyzer.analyze_text(text)
        ai_detection = self.ai_remover.detect_ai_tells(text)

        return {
            "linguistic_features": linguistic,
            "stylometric_features": stylometric,
            "ai_detection": ai_detection,
            "word_count": len(text.split()),
        }

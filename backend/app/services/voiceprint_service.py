"""
Main VoicePrint Service.

Orchestrates the full pipeline: build a writing profile from an author's own
samples, then adapt the author's own drafts toward that voice. The heavy
analyzers and embedders are loaded lazily so importing this module does not pull
in torch or spaCy until a method that needs them runs.
"""
import logging
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.models.style_profile import StyleProfile, profile_store

logger = logging.getLogger(__name__)


class VoicePrintService:
    """Builds writing profiles and adapts drafts toward an author's voice."""

    def __init__(self, spacy_model: str = "en_core_web_sm", store=None):
        """
        Set up the service.

        Args:
            spacy_model: spaCy pipeline name for the analyzers.
            store: Optional profile store. Defaults to the in-memory singleton;
                pass a SqliteProfileStore to persist across restarts.
        """
        self.spacy_model = spacy_model
        self.store = store if store is not None else profile_store

        # All analyzers and embedders are built on first use.
        self._linguistic_analyzer = None
        self._stylometric_analyzer = None
        self._embedding_analyzer = None
        self._style_embedder = None
        self._validator = None
        self._rule_rewriter = None

    # ----- lazy component getters -------------------------------------------

    @property
    def linguistic_analyzer(self):
        if self._linguistic_analyzer is None:
            from app.services.linguistic_analyzer import LinguisticAnalyzer

            self._linguistic_analyzer = LinguisticAnalyzer(self.spacy_model)
        return self._linguistic_analyzer

    @property
    def stylometric_analyzer(self):
        if self._stylometric_analyzer is None:
            from app.services.stylometric_analyzer import StylometricAnalyzer

            self._stylometric_analyzer = StylometricAnalyzer(self.spacy_model)
        return self._stylometric_analyzer

    @property
    def embedding_analyzer(self):
        if self._embedding_analyzer is None:
            from app.services.embedding_analyzer import EmbeddingAnalyzer

            self._embedding_analyzer = EmbeddingAnalyzer(spacy_model=self.spacy_model)
        return self._embedding_analyzer

    @property
    def style_embedder(self):
        if self._style_embedder is None:
            from app.services.style_embedding import StyleEmbedder

            settings = get_settings()
            self._style_embedder = StyleEmbedder(settings.style_model)
        return self._style_embedder

    @property
    def validator(self):
        if self._validator is None:
            from app.services.validator import OutputValidator

            self._validator = OutputValidator(
                self.linguistic_analyzer, self.embedding_analyzer
            )
        return self._validator

    @property
    def rule_rewriter(self):
        if self._rule_rewriter is None:
            from app.services.rewrite.rule_rewriter import RuleRewriter

            self._rule_rewriter = RuleRewriter()
        return self._rule_rewriter

    # ----- profile building --------------------------------------------------

    def build_profile(
        self,
        user_id: str,
        samples: List[str],
        profile_name: Optional[str] = None,
    ) -> StyleProfile:
        """
        Build a writing profile from the author's own samples.

        Validates at least three samples and roughly 500 words total, runs the
        interpretable analyzers, builds the semantic centroid (for meaning
        checks) and the StyleDistance voice centroid (the fingerprint), fits the
        char n-gram profiler, and saves the profile.
        """
        cleaned = [s for s in samples if s and s.strip()]
        if len(cleaned) < 3:
            raise ValueError("Please provide at least 3 writing samples.")

        total_words = sum(len(s.split()) for s in cleaned)
        if total_words < 500:
            raise ValueError(
                f"Need at least 500 words total. You provided {total_words} words."
            )

        logger.info("Building profile for user %s", user_id)
        profile = StyleProfile(
            user_id=user_id, samples=cleaned, profile_name=profile_name
        )

        logger.info("Extracting linguistic features")
        profile.set_linguistic_features(
            self.linguistic_analyzer.analyze_multiple_samples(cleaned)
        )

        logger.info("Extracting stylometric features")
        profile.set_stylometric_features(
            self.stylometric_analyzer.analyze_multiple_samples(cleaned)
        )

        logger.info("Building semantic centroid")
        semantic_centroid = self.embedding_analyzer.create_style_centroid(cleaned)
        consistency = self.embedding_analyzer.analyze_style_consistency(cleaned)
        profile.set_style_centroid(
            semantic_centroid,
            {
                "embedding_dimension": int(len(semantic_centroid)),
                "consistency_metrics": consistency,
            },
        )

        logger.info("Building voice fingerprint centroid")
        voice_centroid = self.style_embedder.build_centroid(cleaned)
        profile.set_voice_centroid(voice_centroid)

        logger.info("Fitting char n-gram profiler")
        from app.services.char_ngram import CharNgramProfiler

        profiler = CharNgramProfiler().fit(cleaned)
        profile.set_char_ngram_profiler(profiler)

        self.store.save_profile(profile)
        logger.info("Profile created: %s", profile.profile_id)
        return profile

    # ----- draft adaptation --------------------------------------------------

    def adapt_draft(
        self,
        profile_id: str,
        source_draft: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Adapt the author's own draft toward their voice.

        Computes the voice match before, rewrites with the LLM rewriter when a
        key is available (otherwise the rule rewriter), computes the voice match
        after, validates meaning preservation against the source draft, and
        builds the radar plus per-sentence feedback.
        """
        profile = self.store.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")

        if not source_draft or len(source_draft.strip()) < 10:
            raise ValueError("Please provide a draft of at least 10 characters.")

        # Refit the char n-gram profiler if the profile was loaded from disk.
        if profile.char_ngram_profiler is None and profile.samples:
            from app.services.char_ngram import CharNgramProfiler

            profile.set_char_ngram_profiler(
                CharNgramProfiler().fit(profile.samples)
            )

        voice_centroid = profile.get_style_centroid_array()
        voice_match_before = (
            self.style_embedder.voice_match(source_draft, voice_centroid)
            if voice_centroid is not None
            else 0.0
        )

        adapted_text, rewrite_path = self._run_rewrite(
            profile, source_draft, api_key, base_url, model, use_llm
        )

        voice_match_after = (
            self.style_embedder.voice_match(adapted_text, voice_centroid)
            if voice_centroid is not None
            else 0.0
        )

        validation = self.validator.validate_output(
            source_draft, adapted_text, profile
        )

        from app.services.feedback import build_feedback

        feedback = build_feedback(
            source_draft=source_draft,
            adapted_text=adapted_text,
            profile=profile,
            style_embedder=self.style_embedder,
            linguistic_analyzer=self.linguistic_analyzer,
            stylometric_analyzer=self.stylometric_analyzer,
            nlp=self.linguistic_analyzer.nlp,
        )

        return {
            "source_draft": source_draft,
            "adapted_text": adapted_text,
            "voice_match_before": voice_match_before,
            "voice_match_after": voice_match_after,
            "rewrite_path": rewrite_path,
            "validation": validation,
            "feedback": feedback,
        }

    def _run_rewrite(
        self,
        profile: StyleProfile,
        source_draft: str,
        api_key: Optional[str],
        base_url: Optional[str],
        model: Optional[str],
        use_llm: bool,
    ):
        """Try the LLM rewriter, fall back to the rule rewriter. Return (text, path)."""
        if use_llm:
            from app.services.rewrite.llm_rewriter import LlmRewriter, NoApiKeyError

            settings = get_settings()
            rewriter = LlmRewriter(
                api_key=api_key or settings.openai_api_key or None,
                base_url=base_url or settings.openai_base_url or None,
                model=model or settings.openai_model or None,
            )
            if rewriter.is_available():
                try:
                    return rewriter.rewrite(source_draft, profile), "llm"
                except (NoApiKeyError, RuntimeError) as exc:
                    logger.warning("LLM rewrite failed, using rule rewriter: %s", exc)

        return self.rule_rewriter.rewrite(source_draft, profile), "rule"

    # ----- profile management ------------------------------------------------

    def get_profile(self, profile_id: str) -> Dict[str, Any]:
        """Get profile details by ID."""
        profile = self.store.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")
        return profile.to_dict()

    def list_user_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all profiles for a user."""
        return [p.to_dict() for p in self.store.get_user_profiles(user_id)]

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile."""
        return self.store.delete_profile(profile_id)

    # ----- quick analysis ----------------------------------------------------

    def quick_analysis(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text with no profile: linguistic and stylometric
        features plus the normalized voice-fingerprint axes.
        """
        from app.services.feedback import normalize_axes

        linguistic = self.linguistic_analyzer.analyze_text(text)
        stylometric = self.stylometric_analyzer.analyze_text(text)
        fingerprint_axes = normalize_axes(linguistic, stylometric)

        return {
            "linguistic_features": linguistic,
            "stylometric_features": stylometric,
            "fingerprint_axes": fingerprint_axes,
            "word_count": len(text.split()),
        }

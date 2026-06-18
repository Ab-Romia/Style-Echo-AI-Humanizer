"""
Style embedding with StyleDistance.

Builds the author's voice fingerprint with the StyleDistance/styledistance
model. StyleDistance is content-independent: two texts about different topics
written in the same voice land near each other, which is what a writing
fingerprint needs. The headline "voice match" is the cosine similarity between
a candidate text and the author's style centroid.

The model is loaded lazily on first use and cached on the instance so that
importing this module never loads torch. If the model cannot load, the methods
raise a clear RuntimeError rather than returning a fabricated score.
"""
from typing import List, Optional, Tuple

import numpy as np


class StyleEmbedder:
    """Loads StyleDistance lazily and scores texts against a style centroid."""

    DEFAULT_MODEL = "StyleDistance/styledistance"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Args:
            model_name: sentence-transformers model id for the style embedder.
        """
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazily load and cache the SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for the style embedder."
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load the style model '{self.model_name}'. "
                    "Check the model id and that it downloaded correctly. "
                    f"Underlying error: {exc}"
                ) from exc
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: Texts to embed.

        Returns:
            Array of shape (len(texts), dim). Empty input gives an empty array.
        """
        if not texts:
            return np.array([])
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True)

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """L2-normalize each row, leaving zero rows untouched."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def build_centroid(self, samples: List[str]) -> np.ndarray:
        """
        Build the author's style centroid.

        Each sample is embedded and L2-normalized, the normalized vectors are
        averaged, and the mean is L2-normalized again so cosine reduces to a
        dot product.

        Args:
            samples: The author's writing samples.

        Returns:
            A unit-length centroid vector.

        Raises:
            ValueError: When no samples are provided.
        """
        if not samples:
            raise ValueError("Cannot build a style centroid from zero samples.")
        embeddings = self.embed(samples)
        if embeddings.size == 0:
            raise ValueError("Embedding produced no vectors.")
        normalized = self._normalize_rows(embeddings)
        centroid = normalized.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return centroid
        return centroid / norm

    def voice_match(self, text: str, centroid: np.ndarray) -> float:
        """
        Cosine similarity between a text and the centroid, clamped to [0, 1].

        Args:
            text: Candidate text.
            centroid: The author's style centroid.

        Returns:
            Voice match in [0, 1].
        """
        if centroid is None or np.size(centroid) == 0:
            return 0.0
        embedding = self.embed([text])
        if embedding.size == 0:
            return 0.0
        vector = self._normalize_rows(embedding)[0]
        centroid = np.asarray(centroid, dtype=float)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            return 0.0
        cosine = float(np.dot(vector, centroid / centroid_norm))
        # Cosine ranges in [-1, 1]; clamp to a [0, 1] match score.
        return max(0.0, min(1.0, cosine))

    def per_sentence_match(
        self, text: str, centroid: np.ndarray, nlp=None
    ) -> List[Tuple[str, float]]:
        """
        Voice match for each sentence in the text.

        Args:
            text: Candidate text.
            centroid: The author's style centroid.
            nlp: Optional loaded spaCy pipeline used to split sentences. When
                absent, a simple punctuation split is used.

        Returns:
            List of (sentence, match) pairs in document order.
        """
        sentences = self._split_sentences(text, nlp)
        if not sentences:
            return []
        embeddings = self.embed(sentences)
        if embeddings.size == 0:
            return []
        normalized = self._normalize_rows(embeddings)
        centroid = np.asarray(centroid, dtype=float)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            return [(sent, 0.0) for sent in sentences]
        unit_centroid = centroid / centroid_norm

        results: List[Tuple[str, float]] = []
        for sentence, vector in zip(sentences, normalized):
            cosine = float(np.dot(vector, unit_centroid))
            results.append((sentence, max(0.0, min(1.0, cosine))))
        return results

    @staticmethod
    def _split_sentences(text: str, nlp=None) -> List[str]:
        """Split text into sentences, preferring spaCy when available."""
        if nlp is not None:
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        import re

        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

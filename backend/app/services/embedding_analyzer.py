"""
Sentence-BERT Embedding and Style Centroid Generation Module.

Uses Sentence-BERT to generate embeddings for text samples and creates
a style centroid vector representing the user's semantic writing patterns.
"""
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingAnalyzer:
    """Generates and analyzes embeddings for style profiling."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding analyzer.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate Sentence-BERT embeddings for a list of texts.

        Args:
            texts: List of text samples

        Returns:
            NumPy array of embeddings, shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def create_style_centroid(self, samples: List[str]) -> np.ndarray:
        """
        Create a style centroid by averaging embeddings from multiple samples.

        Args:
            samples: List of text samples representing user's writing style

        Returns:
            NumPy array representing the style centroid vector
        """
        embeddings = self.generate_embeddings(samples)

        if len(embeddings) == 0:
            return np.zeros(self.embedding_dim)

        # Average all embeddings to create centroid
        centroid = np.mean(embeddings, axis=0)

        # Normalize the centroid
        centroid = centroid / np.linalg.norm(centroid)

        return centroid

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        # Reshape for sklearn if needed
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)

        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return float(similarity)

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two text strings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0 to 1)
        """
        emb1 = self.model.encode(text1, convert_to_numpy=True)
        emb2 = self.model.encode(text2, convert_to_numpy=True)

        return self.compute_similarity(emb1, emb2)

    def compute_centroid_similarity(self, text: str, centroid: np.ndarray) -> float:
        """
        Compute similarity between a text and a style centroid.

        Args:
            text: Input text to compare
            centroid: Style centroid vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        text_embedding = self.model.encode(text, convert_to_numpy=True)
        return self.compute_similarity(text_embedding, centroid)

    def analyze_style_consistency(self, samples: List[str]) -> Dict[str, Any]:
        """
        Analyze how consistent the writing style is across samples.

        Args:
            samples: List of text samples

        Returns:
            Dictionary with consistency metrics
        """
        if len(samples) < 2:
            return {
                "mean_pairwise_similarity": 0.0,
                "std_pairwise_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
            }

        embeddings = self.generate_embeddings(samples)

        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self.compute_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        if not similarities:
            return {
                "mean_pairwise_similarity": 0.0,
                "std_pairwise_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
            }

        return {
            "mean_pairwise_similarity": float(np.mean(similarities)),
            "std_pairwise_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
        }

    def get_sentence_embeddings(self, text: str) -> List[np.ndarray]:
        """
        Get embeddings for individual sentences in a text.

        Args:
            text: Input text

        Returns:
            List of embedding vectors, one per sentence
        """
        # Simple sentence splitting (can be improved with spaCy)
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if not sentences:
            return []

        embeddings = self.generate_embeddings(sentences)
        return [emb for emb in embeddings]

    def find_most_similar_sentences(
        self, target_text: str, candidate_texts: List[str], top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar sentences to a target text.

        Args:
            target_text: Text to compare against
            candidate_texts: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of tuples (text, similarity_score) sorted by similarity
        """
        target_embedding = self.model.encode(target_text, convert_to_numpy=True)
        candidate_embeddings = self.generate_embeddings(candidate_texts)

        similarities = []
        for i, candidate_emb in enumerate(candidate_embeddings):
            sim = self.compute_similarity(target_embedding, candidate_emb)
            similarities.append((candidate_texts[i], sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

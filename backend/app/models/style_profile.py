"""
Style Profile Model.

Represents a user's writing style profile with all extracted features.
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


class StyleProfile:
    """
    Represents a complete style profile for a user.

    Stores linguistic features, stylometric markers, and embedding centroid.
    """

    def __init__(
        self,
        user_id: str,
        samples: List[str],
        profile_name: Optional[str] = None,
        profile_id: Optional[str] = None,
    ):
        """
        Initialize a style profile.

        Args:
            user_id: User identifier
            samples: List of text samples
            profile_name: Optional name for this profile
            profile_id: Optional ID (generated if not provided)
        """
        self.profile_id = profile_id or str(uuid.uuid4())
        self.user_id = user_id
        self.profile_name = profile_name
        self.samples = samples
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        # Features (populated by analyzers)
        self.linguistic_features: Dict[str, Any] = {}
        self.stylometric_features: Dict[str, Any] = {}
        # Semantic centroid (MiniLM); used for meaning preservation checks.
        self.style_centroid: Optional[np.ndarray] = None
        self.embedding_metadata: Dict[str, Any] = {}
        # Voice fingerprint centroid (StyleDistance); the headline voice signal.
        self.voice_centroid: Optional[np.ndarray] = None
        # Fitted char n-gram profiler for this author (not JSON serialized).
        self.char_ngram_profiler: Any = None

    def set_linguistic_features(self, features: Dict[str, Any]) -> None:
        """Set linguistic features from analysis."""
        self.linguistic_features = features
        self.updated_at = datetime.utcnow()

    def set_stylometric_features(self, features: Dict[str, Any]) -> None:
        """Set stylometric features from analysis."""
        self.stylometric_features = features
        self.updated_at = datetime.utcnow()

    def set_style_centroid(self, centroid: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Set the semantic centroid and embedding metadata."""
        self.style_centroid = centroid
        self.embedding_metadata = metadata
        self.updated_at = datetime.utcnow()

    def set_voice_centroid(self, centroid: np.ndarray) -> None:
        """Set the StyleDistance voice fingerprint centroid."""
        self.voice_centroid = centroid
        self.updated_at = datetime.utcnow()

    def set_char_ngram_profiler(self, profiler: Any) -> None:
        """Attach the fitted char n-gram profiler for this author."""
        self.char_ngram_profiler = profiler
        self.updated_at = datetime.utcnow()

    def get_style_centroid_array(self) -> Optional[np.ndarray]:
        """Return the voice fingerprint centroid for scoring, if present."""
        return self.voice_centroid

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary (for JSON serialization)."""
        return {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "profile_name": self.profile_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "num_samples": len(self.samples),
            "total_words": self.linguistic_features.get("total_word_count", 0),
            "linguistic_features": self.linguistic_features,
            "stylometric_features": self.stylometric_features,
            "embedding_metadata": self.embedding_metadata,
            "has_voice_centroid": self.voice_centroid is not None,
            # Note: the numpy centroids are stored separately, not in this dict.
        }

    def get_centroid_as_list(self) -> Optional[List[float]]:
        """Get the semantic centroid as a list (for JSON serialization)."""
        if self.style_centroid is not None:
            return self.style_centroid.tolist()
        return None

    def get_voice_centroid_as_list(self) -> Optional[List[float]]:
        """Get the voice fingerprint centroid as a list."""
        if self.voice_centroid is not None:
            return self.voice_centroid.tolist()
        return None

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        centroid: Optional[np.ndarray] = None,
        voice_centroid: Optional[np.ndarray] = None,
    ) -> "StyleProfile":
        """
        Create a StyleProfile from dictionary.

        Args:
            data: Dictionary with profile data
            centroid: Optional numpy array for the semantic centroid
            voice_centroid: Optional numpy array for the voice fingerprint

        Returns:
            StyleProfile instance
        """
        profile = cls(
            user_id=data["user_id"],
            samples=data.get("samples", []),
            profile_name=data.get("profile_name"),
            profile_id=data.get("profile_id"),
        )

        profile.linguistic_features = data.get("linguistic_features", {})
        profile.stylometric_features = data.get("stylometric_features", {})
        profile.embedding_metadata = data.get("embedding_metadata", {})

        if centroid is not None:
            profile.style_centroid = centroid
        if voice_centroid is not None:
            profile.voice_centroid = voice_centroid

        if "created_at" in data:
            profile.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            profile.updated_at = datetime.fromisoformat(data["updated_at"])

        return profile


class StyleProfileStore:
    """
    In-memory store for style profiles.

    Suitable for a single-process demo. For persistence across restarts, the
    SQLite-backed store in app.store.sqlite_store exposes the same interface.
    """

    def __init__(self):
        """Initialize the profile store."""
        self.profiles: Dict[str, StyleProfile] = {}
        self.user_profiles: Dict[str, List[str]] = {}  # user_id -> [profile_ids]

    def save_profile(self, profile: StyleProfile) -> None:
        """
        Save a profile to the store.

        Args:
            profile: StyleProfile to save
        """
        self.profiles[profile.profile_id] = profile

        # Update user index
        if profile.user_id not in self.user_profiles:
            self.user_profiles[profile.user_id] = []

        if profile.profile_id not in self.user_profiles[profile.user_id]:
            self.user_profiles[profile.user_id].append(profile.profile_id)

    def get_profile(self, profile_id: str) -> Optional[StyleProfile]:
        """
        Get a profile by ID.

        Args:
            profile_id: Profile identifier

        Returns:
            StyleProfile if found, None otherwise
        """
        return self.profiles.get(profile_id)

    def get_user_profiles(self, user_id: str) -> List[StyleProfile]:
        """
        Get all profiles for a user.

        Args:
            user_id: User identifier

        Returns:
            List of StyleProfiles
        """
        profile_ids = self.user_profiles.get(user_id, [])
        return [self.profiles[pid] for pid in profile_ids if pid in self.profiles]

    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a profile.

        Args:
            profile_id: Profile identifier

        Returns:
            True if deleted, False if not found
        """
        if profile_id not in self.profiles:
            return False

        profile = self.profiles[profile_id]
        user_id = profile.user_id

        # Remove from profiles
        del self.profiles[profile_id]

        # Remove from user index
        if user_id in self.user_profiles and profile_id in self.user_profiles[user_id]:
            self.user_profiles[user_id].remove(profile_id)

        return True


# Global instance (in production, use dependency injection)
profile_store = StyleProfileStore()

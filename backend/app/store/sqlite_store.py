"""
SQLite-backed profile store.

A thin, dependency-free persistence layer that exposes the same interface as
the in-memory StyleProfileStore (save_profile, get_profile, get_user_profiles,
delete_profile). Features are stored as JSON; the two numpy centroids are
stored as raw float32 bytes so they round-trip without precision loss.

The char n-gram profiler is not serialized: it is cheap to refit from the
stored samples when a profile is loaded, which keeps the store free of any
pickle dependency.
"""
import json
import os
import sqlite3
from datetime import datetime
from typing import List, Optional

import numpy as np

from app.models.style_profile import StyleProfile


DEFAULT_DB_PATH = os.path.join("data", "voiceprint.db")


def _to_bytes(array: Optional[np.ndarray]) -> Optional[bytes]:
    """Serialize a numpy array to float32 bytes, or None."""
    if array is None:
        return None
    return np.asarray(array, dtype=np.float32).tobytes()


def _from_bytes(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """Deserialize float32 bytes back to a numpy array, or None."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


class SqliteProfileStore:
    """Persists style profiles in a single SQLite file."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Args:
            db_path: Path to the SQLite database file. The parent directory is
                created if it does not exist. Defaults to data/voiceprint.db,
                under a gitignored data dir.
        """
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    profile_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    profile_name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    samples TEXT NOT NULL,
                    linguistic_features TEXT NOT NULL,
                    stylometric_features TEXT NOT NULL,
                    embedding_metadata TEXT NOT NULL,
                    style_centroid BLOB,
                    voice_centroid BLOB
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_profiles_user ON profiles(user_id)"
            )

    def save_profile(self, profile: StyleProfile) -> None:
        """Insert or replace a profile."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO profiles (
                    profile_id, user_id, profile_name, created_at, updated_at,
                    samples, linguistic_features, stylometric_features,
                    embedding_metadata, style_centroid, voice_centroid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.profile_id,
                    profile.user_id,
                    profile.profile_name,
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                    json.dumps(profile.samples),
                    json.dumps(profile.linguistic_features),
                    json.dumps(profile.stylometric_features),
                    json.dumps(profile.embedding_metadata),
                    _to_bytes(profile.style_centroid),
                    _to_bytes(profile.voice_centroid),
                ),
            )

    def _row_to_profile(self, row: sqlite3.Row) -> StyleProfile:
        profile = StyleProfile(
            user_id=row["user_id"],
            samples=json.loads(row["samples"]),
            profile_name=row["profile_name"],
            profile_id=row["profile_id"],
        )
        profile.linguistic_features = json.loads(row["linguistic_features"])
        profile.stylometric_features = json.loads(row["stylometric_features"])
        profile.embedding_metadata = json.loads(row["embedding_metadata"])
        profile.style_centroid = _from_bytes(row["style_centroid"])
        profile.voice_centroid = _from_bytes(row["voice_centroid"])
        profile.created_at = datetime.fromisoformat(row["created_at"])
        profile.updated_at = datetime.fromisoformat(row["updated_at"])
        return profile

    def get_profile(self, profile_id: str) -> Optional[StyleProfile]:
        """Return a profile by id, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_profile(row)

    def get_user_profiles(self, user_id: str) -> List[StyleProfile]:
        """Return all profiles for a user, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM profiles WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [self._row_to_profile(row) for row in rows]

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile by id. Return True if a row was removed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM profiles WHERE profile_id = ?", (profile_id,)
            )
            return cursor.rowcount > 0

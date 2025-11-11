"""
Pydantic schemas for style profiles and related data.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class TextSampleInput(BaseModel):
    """Schema for text sample input."""

    text: str = Field(..., min_length=50, description="Text sample (minimum 50 characters)")


class StyleProfileCreate(BaseModel):
    """Schema for creating a new style profile."""

    user_id: str = Field(..., description="User identifier")
    samples: List[str] = Field(..., min_items=3, max_items=10, description="3-10 text samples")
    profile_name: Optional[str] = Field(None, description="Optional name for this profile")


class StyleProfileResponse(BaseModel):
    """Schema for style profile response."""

    profile_id: str
    user_id: str
    profile_name: Optional[str]
    created_at: datetime
    updated_at: datetime
    total_words: int
    num_samples: int
    linguistic_features: Dict[str, Any]
    stylometric_features: Dict[str, Any]
    embedding_metadata: Dict[str, Any]


class HumanizeRequest(BaseModel):
    """Schema for text humanization request."""

    profile_id: str = Field(..., description="Style profile ID to use")
    text: str = Field(..., min_length=10, description="AI-generated text to humanize")
    strength: float = Field(0.7, ge=0.0, le=1.0, description="Humanization strength (0-1)")
    preserve_meaning: bool = Field(True, description="Whether to preserve semantic meaning")


class HumanizeResponse(BaseModel):
    """Schema for humanization response."""

    original_text: str
    humanized_text: str
    similarity_score: float
    ai_detection_score: Optional[float]
    transformation_metadata: Dict[str, Any]


class ValidationMetrics(BaseModel):
    """Schema for validation metrics."""

    style_similarity: float
    semantic_preservation: float
    readability_match: bool
    linguistic_feature_match: Dict[str, float]
    overall_quality_score: float

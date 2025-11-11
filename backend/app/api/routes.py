"""
API Routes for VoicePrint.

Endpoints for profile creation, text humanization, and profile management.
"""
from fastapi import APIRouter, HTTPException, status
from typing import List
import logging

from app.schemas.profile import (
    StyleProfileCreate,
    StyleProfileResponse,
    HumanizeRequest,
    HumanizeResponse,
    TextSampleInput,
)
from app.services.voiceprint_service import VoicePrintService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize the main service
voiceprint_service = VoicePrintService()


@router.post("/profiles", response_model=StyleProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile(request: StyleProfileCreate):
    """
    Create a new writing style profile.

    Upload 3-10 text samples (min 500 words total) to build your profile.
    """
    try:
        profile = voiceprint_service.create_style_profile(
            user_id=request.user_id,
            samples=request.samples,
            profile_name=request.profile_name,
        )

        return StyleProfileResponse(**profile.to_dict())

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while building your profile"
        )


@router.get("/profiles/{profile_id}", response_model=StyleProfileResponse)
async def get_profile(profile_id: str):
    """Get details of a specific style profile."""
    try:
        profile_data = voiceprint_service.get_profile(profile_id)
        return StyleProfileResponse(**profile_data)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't retrieve that profile"
        )


@router.get("/users/{user_id}/profiles", response_model=List[StyleProfileResponse])
async def list_user_profiles(user_id: str):
    """Get all style profiles for a user."""
    try:
        profiles = voiceprint_service.list_user_profiles(user_id)
        return [StyleProfileResponse(**p) for p in profiles]

    except Exception as e:
        logger.error(f"Error listing profiles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't load your profiles"
        )


@router.delete("/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(profile_id: str):
    """Delete a style profile."""
    try:
        success = voiceprint_service.delete_profile(profile_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't delete that profile"
        )


@router.post("/humanize", response_model=HumanizeResponse)
async def humanize_text(request: HumanizeRequest):
    """
    Transform AI-generated text into your writing style.

    Takes AI text and makes it sound like you wrote it.
    """
    try:
        result = voiceprint_service.humanize_text(
            profile_id=request.profile_id,
            ai_text=request.text,
            strength=request.strength,
            preserve_meaning=request.preserve_meaning,
        )

        return HumanizeResponse(
            original_text=result["original_text"],
            humanized_text=result["humanized_text"],
            similarity_score=result["validation"]["style_similarity"],
            ai_detection_score=result["ai_removal_metrics"]["improved_ai_score"],
            transformation_metadata=result["metadata"],
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error humanizing text: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong during humanization"
        )


@router.post("/analyze")
async def quick_analyze(request: TextSampleInput):
    """
    Quick analysis of text without creating a profile.

    Good for testing what features we extract from your writing.
    """
    try:
        analysis = voiceprint_service.quick_analysis(request.text)
        return analysis

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "running", "service": "VoicePrint"}

"""
API Routes for VoicePrint.

Endpoints for profile creation, draft adaptation, and profile management. The
service is instantiated lazily through a dependency so importing this module
never loads the heavy models.
"""
import logging
from functools import lru_cache
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.profile import (
    AdaptRequest,
    AdaptResponse,
    StyleProfileCreate,
    StyleProfileResponse,
    TextSampleInput,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache()
def _build_service():
    """Build the service once, on first request, not at import time."""
    from app.config import get_settings
    from app.services.voiceprint_service import VoicePrintService
    from app.store.sqlite_store import SqliteProfileStore

    settings = get_settings()
    store = SqliteProfileStore(settings.profile_db_path)
    return VoicePrintService(spacy_model=settings.spacy_model, store=store)


def get_service():
    """FastAPI dependency that returns the shared service instance."""
    return _build_service()


@router.post(
    "/profiles",
    response_model=StyleProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_profile(request: StyleProfileCreate, service=Depends(get_service)):
    """
    Build a writing profile from your own samples.

    Provide 3 to 10 samples (at least 500 words total) to measure your voice.
    """
    try:
        profile = service.build_profile(
            user_id=request.user_id,
            samples=request.samples,
            profile_name=request.profile_name,
        )
        return StyleProfileResponse(**profile.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Error creating profile: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while building your profile",
        )


@router.get("/profiles/{profile_id}", response_model=StyleProfileResponse)
async def get_profile(profile_id: str, service=Depends(get_service)):
    """Get details of a specific profile."""
    try:
        return StyleProfileResponse(**service.get_profile(profile_id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error("Error fetching profile: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't retrieve that profile",
        )


@router.get("/users/{user_id}/profiles", response_model=List[StyleProfileResponse])
async def list_user_profiles(user_id: str, service=Depends(get_service)):
    """Get all profiles for a user."""
    try:
        return [StyleProfileResponse(**p) for p in service.list_user_profiles(user_id)]
    except Exception as e:
        logger.error("Error listing profiles: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't load your profiles",
        )


@router.delete("/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(profile_id: str, service=Depends(get_service)):
    """Delete a profile."""
    try:
        if not service.delete_profile(profile_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting profile: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't delete that profile",
        )


@router.post("/adapt", response_model=AdaptResponse)
async def adapt_draft(request: AdaptRequest, service=Depends(get_service)):
    """
    Adapt your own draft toward your measured writing voice.

    Uses the LLM rewriter when an API key is available, otherwise the
    rule-based rewriter.
    """
    try:
        result = service.adapt_draft(
            profile_id=request.profile_id,
            source_draft=request.source_draft,
            api_key=request.api_key,
            base_url=request.base_url,
            model=request.model,
            use_llm=request.use_llm,
        )
        return AdaptResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Error adapting draft: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while adapting your draft",
        )


@router.post("/analyze")
async def quick_analyze(request: TextSampleInput, service=Depends(get_service)):
    """Analyze a single text without building a profile."""
    try:
        return service.quick_analysis(request.text)
    except Exception as e:
        logger.error("Error analyzing text: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed",
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "running", "service": "VoicePrint"}

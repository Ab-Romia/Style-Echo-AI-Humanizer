"""
FastAPI application entry point.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

settings = get_settings()

app = FastAPI(
    title="VoicePrint API",
    description="Measure your writing voice and adapt your drafts toward it",
    version="0.1.0",
)

# CORS is locked to known origins. Add the deployed frontend origin here once
# its URL is confirmed; do not widen this to "*".
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:7860",
    # "https://your-frontend-domain.example",  # set to the real origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["voiceprint"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to VoicePrint API",
        "docs": "/docs",
        "version": "0.1.0"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )

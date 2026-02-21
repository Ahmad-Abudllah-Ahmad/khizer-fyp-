from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from contextlib import asynccontextmanager
from app.models.unified_model import UnifiedMentalHealthAnalyzer
from app.routers import analyze, health
from app.core.config import settings
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global model instance ───────────────────────────────────────────────────
unified_analyzer: UnifiedMentalHealthAnalyzer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global unified_analyzer

    logger.info(f"🚀 Starting {settings.PROJECT_NAME} ...")

    if not os.path.exists(settings.UNIFIED_MODEL_PATH):
        logger.critical(f"❌ Model not found at {settings.UNIFIED_MODEL_PATH}")
        raise RuntimeError(f"Unified model file not found: {settings.UNIFIED_MODEL_PATH}")

    logger.info("📦 Loading Unified Mental Health Model v4 ...")
    unified_analyzer = UnifiedMentalHealthAnalyzer(model_path=settings.UNIFIED_MODEL_PATH)
    logger.info("✅ Unified model loaded — full semantic analysis active")

    yield
    logger.info("🛑 Shutting down AI service ...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="SereneMind AI — Unified mental health analysis (emotion · crisis · severity · tags)",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,  prefix="/health",  tags=["Health"])
app.include_router(analyze.router, prefix="/analyze", tags=["Analysis"])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=2)

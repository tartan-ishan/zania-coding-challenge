import logging
import sys

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    # Fail fast — validates OPENAI_API_KEY is present at startup
    settings = get_settings()
    logger.info("Starting QA service with model=%s", settings.openai_model)

    app = FastAPI(
        title="Zania QA API",
        description="Answer questions from PDF and JSON documents using RAG.",
        version="1.0.0",
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()

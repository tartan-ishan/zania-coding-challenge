import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    # Fail fast — validates OPENAI_API_KEY is present at startup
    settings = get_settings()
    logger.info("starting", extra={"model": settings.openai_model})

    app = FastAPI(
        title="Zania QA API",
        description="Answer questions from PDF and JSON documents using RAG.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        logger.info("#" * 60)
        return response

    @app.get("/health", include_in_schema=False)
    async def health():
        return JSONResponse({"status": "ok"})

    static_dir = Path(__file__).parent.parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(static_dir / "index.html")

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()

import logging
import time

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.config import get_settings
from app.models.schemas import ErrorResponse, QAResponse
from app.services.document_loader import load_documents, load_questions
from app.services.qa_service import answer_questions
from app.services.vector_store import build_retriever

logger = logging.getLogger(__name__)

router = APIRouter()

SUPPORTED_DOCUMENT_CONTENT_TYPES = {"application/pdf", "application/json"}


def _error(status_code: int, error_code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(error_code=error_code, message=message).model_dump(),
    )


@router.post(
    "/qa",
    response_model=QAResponse,
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        422: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Document processing error"},
        502: {"model": ErrorResponse, "description": "AI service error"},
    },
    summary="Answer questions from a document",
    description=(
        "Upload a document (PDF or JSON) and a questions file (JSON array of strings). "
        "Returns each question paired with its answer extracted from the document."
    ),
)
async def question_answer(
    document_file: UploadFile = File(..., description="PDF or JSON document to query"),
    questions_file: UploadFile = File(..., description="JSON file containing a list of questions"),
) -> QAResponse:
    request_start = time.perf_counter()
    logger.info(
        "Received request: document=%s (%s), questions=%s (%s)",
        document_file.filename, document_file.content_type,
        questions_file.filename, questions_file.content_type,
    )

    # Validate content types
    if document_file.content_type not in SUPPORTED_DOCUMENT_CONTENT_TYPES:
        raise _error(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "UNSUPPORTED_DOCUMENT_TYPE",
            f"Unsupported document type '{document_file.content_type}'. Upload a PDF or JSON file.",
        )
    if questions_file.content_type != "application/json":
        raise _error(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "UNSUPPORTED_QUESTIONS_TYPE",
            f"Questions file must be JSON, got '{questions_file.content_type}'.",
        )

    settings = get_settings()

    doc_bytes = await document_file.read()
    if len(doc_bytes) > settings.max_document_bytes:
        raise _error(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "DOCUMENT_TOO_LARGE",
            f"Document exceeds the {settings.max_document_bytes // (1024 * 1024)} MB limit.",
        )

    questions_bytes = await questions_file.read()
    if len(questions_bytes) > settings.max_questions_bytes:
        raise _error(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "QUESTIONS_FILE_TOO_LARGE",
            f"Questions file exceeds the {settings.max_questions_bytes // 1024} KB limit.",
        )

    # Parse inputs
    try:
        documents = load_documents(doc_bytes, document_file.content_type)
    except ValueError as e:
        logger.warning("Document parsing failed: %s", e)
        raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, "INVALID_DOCUMENT", str(e))

    try:
        questions = load_questions(questions_bytes, questions_file.content_type)
    except ValueError as e:
        logger.warning("Questions parsing failed: %s", e)
        raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, "INVALID_QUESTIONS", str(e))

    if len(questions) > settings.max_questions:
        raise _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "TOO_MANY_QUESTIONS",
            f"Too many questions: {len(questions)} submitted, maximum is {settings.max_questions}.",
        )

    logger.info("Parsed %d document chunks, %d questions", len(documents), len(questions))

    # Build hybrid retriever
    try:
        retriever = build_retriever(documents)
    except Exception as e:
        logger.error("Failed to build retriever: %s", e, exc_info=True)
        raise _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "RETRIEVER_BUILD_FAILED",
            "Failed to process the document. Please check the file and try again.",
        )

    # Answer questions
    try:
        answers = await answer_questions(questions, retriever)
    except Exception as e:
        logger.error("Failed to answer questions: %s", e, exc_info=True)
        raise _error(
            status.HTTP_502_BAD_GATEWAY,
            "AI_SERVICE_ERROR",
            "The AI service is temporarily unavailable. Please try again.",
        )

    latency_ms = round((time.perf_counter() - request_start) * 1000, 2)
    logger.info(
        "Request complete: answered %d questions",
        len(answers),
        extra={"latency_ms": latency_ms},
    )
    return QAResponse(answers=answers)

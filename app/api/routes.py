import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.schemas import QAResponse
from app.services.document_loader import load_documents, load_questions
from app.services.qa_service import answer_questions
from app.services.vector_store import build_retriever

logger = logging.getLogger(__name__)

router = APIRouter()

SUPPORTED_DOCUMENT_CONTENT_TYPES = {"application/pdf", "application/json"}


@router.post(
    "/qa",
    response_model=QAResponse,
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
    # Validate content types
    if document_file.content_type not in SUPPORTED_DOCUMENT_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported document type '{document_file.content_type}'. "
                "Upload a PDF or JSON file."
            ),
        )
    if questions_file.content_type != "application/json":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Questions file must be JSON, got '{questions_file.content_type}'."
            ),
        )

    doc_bytes = await document_file.read()
    questions_bytes = await questions_file.read()

    # Parse inputs
    try:
        documents = load_documents(doc_bytes, document_file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))

    try:
        questions = load_questions(questions_bytes, questions_file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))

    # Build hybrid retriever
    try:
        retriever = build_retriever(documents)
    except Exception as e:
        logger.error("Failed to build retriever: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document. Please try again.",
        )

    # Answer questions
    try:
        answers = await answer_questions(questions, retriever)
    except Exception as e:
        logger.error("Failed to answer questions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM service error: {e}",
        )

    return QAResponse(answers=answers)

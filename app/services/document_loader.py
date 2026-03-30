import json
import logging
import re

import fitz  # PyMuPDF
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)

SUPPORTED_DOC_TYPES = {"application/pdf", "application/json"}

# Strip bold markers from header text (e.g. "**Incident Management**" → "Incident Management")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def load_documents(content: bytes, content_type: str) -> list[Document]:
    """Parse and chunk an uploaded document into LangChain Documents."""
    logger.info("Loading document: content_type=%s size=%d bytes", content_type, len(content))
    if content_type == "application/pdf":
        return _load_pdf(content)
    elif content_type == "application/json":
        return _load_json(content)
    else:
        logger.error("Unsupported document content_type: %s", content_type)
        raise ValueError(
            f"Unsupported document type '{content_type}'. "
            f"Supported types: PDF, JSON."
        )


def load_questions(content: bytes, content_type: str) -> list[str]:
    """Parse a questions JSON file into a list of question strings."""
    logger.info("Loading questions: size=%d bytes", len(content))
    if content_type != "application/json":
        logger.error("Invalid questions content_type: %s", content_type)
        raise ValueError(
            f"Questions file must be JSON, got '{content_type}'."
        )
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse questions JSON: %s", e)
        raise ValueError(f"Invalid JSON in questions file: {e}") from e

    if not isinstance(data, list):
        logger.error("Questions file is not a JSON array: got %s", type(data).__name__)
        raise ValueError("Questions file must contain a JSON array of strings.")

    questions = []
    for item in data:
        if not isinstance(item, str):
            logger.error("Non-string question encountered: %s", type(item).__name__)
            raise ValueError(
                f"Each question must be a string, got {type(item).__name__}."
            )
        stripped = item.strip()
        if not stripped:
            logger.warning("Skipping empty or whitespace-only question")
            continue
        questions.append(stripped)

    if not questions:
        logger.error("Questions file is empty")
        raise ValueError("Questions file must contain at least one question.")

    logger.info("Loaded %d questions", len(questions))
    return questions


def _load_pdf(content: bytes) -> list[Document]:
    """
    PDF → Markdown (via pymupdf4llm, preserving headers from font metadata)
    → split on ## section headers
    → sub-split each section into ~1k char chunks with overlap
    → enrich each chunk with section + chunk_index metadata and a header prefix.
    """
    settings = get_settings()

    try:
        pdf_doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        logger.error("Failed to open PDF: %s", e, exc_info=True)
        raise ValueError(f"Could not open PDF — file may be corrupt or password-protected: {e}") from e

    logger.info("Opened PDF: %d pages", pdf_doc.page_count)
    md_text = pymupdf4llm.to_markdown(pdf_doc)

    if not md_text.strip():
        logger.error("PDF produced no extractable text")
        raise ValueError("PDF document appears to be empty or has no extractable text.")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section")],
        strip_headers=False,  # keep header text in chunk for BM25 keyword matching
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "|", " "],  # "|" helps split at markdown table row boundaries
    )

    section_chunks = header_splitter.split_text(md_text)
    if not section_chunks:
        logger.error("PDF produced no chunks after header splitting")
        raise ValueError("PDF produced no chunks after splitting.")

    enriched: list[Document] = []
    for section_chunk in section_chunks:
        section = _BOLD_RE.sub(r"\1", section_chunk.metadata.get("section", "Document")).strip()

        # Sub-split every section so even large table sections get granular chunks
        sub_docs = char_splitter.create_documents(
            [section_chunk.page_content],
            metadatas=[{"source": "pdf", "section": section}],
        )

        for i, sub in enumerate(sub_docs):
            enriched.append(
                Document(
                    page_content=f"[Section: {section}]\n{sub.page_content}",
                    metadata={
                        "source": "pdf",
                        "section": section,
                        "chunk_index": i,
                    },
                )
            )

    logger.info("PDF chunked into %d documents", len(enriched))
    return enriched


def _load_json(content: bytes) -> list[Document]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse document JSON: %s", e)
        raise ValueError(f"Invalid JSON in document file: {e}") from e

    settings = get_settings()
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    leaf_chunks = _json_to_chunks(data)

    if not leaf_chunks:
        logger.error("JSON document produced no leaf chunks")
        raise ValueError("JSON document appears to be empty.")

    docs: list[Document] = []
    chunk_index = 0
    for label, text in leaf_chunks:
        if len(text) > settings.chunk_size:
            for sub in char_splitter.split_text(text):
                docs.append(
                    Document(
                        page_content=f"[Key: {label}]\n{sub}",
                        metadata={"source": "json", "key": label, "chunk_index": chunk_index},
                    )
                )
                chunk_index += 1
        else:
            docs.append(
                Document(
                    page_content=f"{label}: {text}",
                    metadata={"source": "json", "key": label, "chunk_index": chunk_index},
                )
            )
            chunk_index += 1

    logger.info("JSON chunked into %d documents", len(docs))
    return docs


def _json_to_chunks(data: object, prefix: str = "") -> list[tuple[str, str]]:
    """
    Recursively serialize JSON into (label, text) leaf pairs.
    Large values are sub-split by the caller using RecursiveCharacterTextSplitter.
    """
    chunks = []

    if isinstance(data, dict):
        for key, value in data.items():
            label = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                chunks.extend(_json_to_chunks(value, prefix=label))
            else:
                chunks.append((label, str(value)))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            label = f"{prefix}[{i}]" if prefix else f"[{i}]"
            if isinstance(item, (dict, list)):
                chunks.extend(_json_to_chunks(item, prefix=label))
            else:
                chunks.append((label, str(item)))

    else:
        chunks.append((prefix, str(data)))

    return chunks

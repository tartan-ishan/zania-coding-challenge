import json
import re

import fitz  # PyMuPDF
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import get_settings

SUPPORTED_DOC_TYPES = {"application/pdf", "application/json"}

# Strip bold markers from header text (e.g. "**Incident Management**" → "Incident Management")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def load_documents(content: bytes, content_type: str) -> list[Document]:
    """Parse and chunk an uploaded document into LangChain Documents."""
    if content_type == "application/pdf":
        return _load_pdf(content)
    elif content_type == "application/json":
        return _load_json(content)
    else:
        raise ValueError(
            f"Unsupported document type '{content_type}'. "
            f"Supported types: PDF, JSON."
        )


def load_questions(content: bytes, content_type: str) -> list[str]:
    """Parse a questions JSON file into a list of question strings."""
    if content_type != "application/json":
        raise ValueError(
            f"Questions file must be JSON, got '{content_type}'."
        )
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in questions file: {e}") from e

    if not isinstance(data, list):
        raise ValueError("Questions file must contain a JSON array of strings.")

    questions = []
    for item in data:
        if not isinstance(item, str):
            raise ValueError(
                f"Each question must be a string, got {type(item).__name__}."
            )
        questions.append(item)

    if not questions:
        raise ValueError("Questions file must contain at least one question.")

    return questions


def _load_pdf(content: bytes) -> list[Document]:
    """
    PDF → Markdown (via pymupdf4llm, preserving headers from font metadata)
    → split on ## section headers
    → sub-split each section into ~1k char chunks with overlap
    → enrich each chunk with section + chunk_index metadata and a header prefix.
    """
    settings = get_settings()

    pdf_doc = fitz.open(stream=content, filetype="pdf")
    md_text = pymupdf4llm.to_markdown(pdf_doc)

    if not md_text.strip():
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

    return enriched


def _load_json(content: bytes) -> list[Document]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in document file: {e}") from e

    chunks = _json_to_chunks(data)

    if not chunks:
        raise ValueError("JSON document appears to be empty.")

    return [
        Document(
            page_content=chunk,
            metadata={"source": "json", "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]


def _json_to_chunks(data: object, prefix: str = "") -> list[str]:
    """
    Recursively serialize JSON into semantic text chunks.
    Each top-level key-value pair (or array item) becomes its own chunk,
    preserving structure as readable text.
    """
    chunks = []

    if isinstance(data, dict):
        for key, value in data.items():
            label = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                sub_chunks = _json_to_chunks(value, prefix=label)
                chunks.extend(sub_chunks)
            else:
                chunks.append(f"{label}: {value}")

    elif isinstance(data, list):
        for i, item in enumerate(data):
            label = f"{prefix}[{i}]" if prefix else f"[{i}]"
            if isinstance(item, (dict, list)):
                sub_chunks = _json_to_chunks(item, prefix=label)
                chunks.extend(sub_chunks)
            else:
                chunks.append(f"{label}: {item}")

    else:
        chunks.append(f"{prefix}: {data}" if prefix else str(data))

    return chunks

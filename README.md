# Zania QA API

A production-quality backend API that answers questions from uploaded documents (PDF or JSON) using Retrieval-Augmented Generation (RAG).

**Stack:** FastAPI · LangChain · OpenAI gpt-4o-mini · Chroma (in-memory) · uv

---

## How it works

1. You upload a document (PDF or JSON) and a questions file (JSON array of strings).
2. The document is chunked and embedded into an in-memory Chroma vector store.
3. For each question, the most relevant chunks are retrieved. If no chunk meets the confidence threshold, the answer is `"Data Not Available"`.
4. Qualifying chunks are sent to `gpt-4o-mini` with a strict prompt that prevents hallucination.
5. All questions are answered concurrently. Results are returned as a JSON object mapping each question to its answer.

---

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

### Install

```bash
git clone <repo-url>
cd zania-coding-challenge

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

---

## Running locally

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

---

## Running with Docker

```bash
docker build -t zania-qa .
docker run -p 8000:8000 --env-file .env zania-qa
```

---

## API Usage

### `POST /api/v1/qa`

Upload a document and questions file, get answers back.

**Parameters (multipart/form-data):**

| Field | Type | Description |
|---|---|---|
| `document_file` | file | PDF or JSON document to query |
| `questions_file` | file | JSON file — array of question strings |

**Example request:**

```bash
curl -X POST http://localhost:8000/api/v1/qa \
  -F "document_file=@/path/to/document.pdf" \
  -F "questions_file=@/path/to/questions.json"
```

**Example questions file (`questions.json`):**

```json
[
  "What cloud providers are used?",
  "Who is responsible for security incidents?"
]
```

**Example response:**

```json
{
  "answers": {
    "What cloud providers are used?": "AWS and GCP are used.",
    "Who is responsible for security incidents?": "Data Not Available"
  }
}
```

---

## Configuration

All settings can be overridden via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum retrieval score (0–1) to use a chunk |
| `RETRIEVAL_K` | `4` | Number of chunks to retrieve per question |
| `MAX_RETRIES` | `3` | Retries on OpenAI rate limit errors |

---

## Running tests

### Unit tests (no OpenAI calls, no API key needed)

```bash
uv run pytest tests/unit -v
```

### Integration tests (hit real OpenAI API)

Uses fixture files in `tests/fixtures/` by default.

```bash
uv run pytest tests/integration -v
```

Override fixture files with environment variables:

```bash
TEST_DOCUMENT_PATH=/path/to/your.pdf \
TEST_QUESTIONS_PATH=/path/to/your_questions.json \
uv run pytest tests/integration -v
```

### All tests

```bash
uv run pytest -v
```

---

## Extending

**Switch to persistent Chroma:** In [app/services/vector_store.py](app/services/vector_store.py), pass `persist_directory="./chroma_db"` to `Chroma.from_documents()`.

**Switch to Pinecone:** Replace `build_vector_store()` in [app/services/vector_store.py](app/services/vector_store.py) with a Pinecone-backed store implementing LangChain's `VectorStore` interface. The rest of the code is unchanged.

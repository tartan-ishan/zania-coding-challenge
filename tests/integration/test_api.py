"""
Integration tests — no real OpenAI calls required.

All LLM and embedding calls are mocked so these tests run offline,
without OPENAI_API_KEY, and with deterministic responses.

By default, uses the fixture files in tests/fixtures/.
Override with environment variables:
  TEST_DOCUMENT_PATH  — path to a PDF or JSON document
  TEST_QUESTIONS_PATH — path to a JSON questions file
"""
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import StructuredAnswer

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

DEFAULT_DOCUMENT = FIXTURES_DIR / "sample_document.json"
DEFAULT_QUESTIONS = FIXTURES_DIR / "sample_questions.json"


def _get_document_path() -> Path:
    return Path(os.environ.get("TEST_DOCUMENT_PATH", DEFAULT_DOCUMENT))


def _get_questions_path() -> Path:
    return Path(os.environ.get("TEST_QUESTIONS_PATH", DEFAULT_QUESTIONS))


def _content_type(path: Path) -> str:
    if path.suffix == ".pdf":
        return "application/pdf"
    return "application/json"


def _make_fake_retriever():
    """Return a mock retriever that returns a fixed document for any query."""
    from langchain_core.documents import Document

    doc = Document(page_content="Mocked document content relevant to the question.")
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=[doc])
    mock.invoke = MagicMock(return_value=[doc])
    return mock


def _make_fake_llm(answer: str = "Mocked answer from document context"):
    """Return a mock ChatOpenAI that returns a fixed StructuredAnswer via with_structured_output."""
    structured_answer = StructuredAnswer(
        answer=answer,
        stepwise_reasoning=["Step 1: found relevant context."],
        confidence=0.9,
        citations=["Mocked document content relevant to the question."],
    )
    structured_mock = MagicMock()
    structured_mock.ainvoke = AsyncMock(return_value=structured_answer)

    mock = MagicMock()
    mock.with_structured_output = MagicMock(return_value=structured_mock)
    # also support plain ainvoke for decompose/keyword prompts
    response = MagicMock()
    response.content = "sub-query 1"
    mock.ainvoke = AsyncMock(return_value=response)
    return mock


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def qa_response(client):
    doc_path = _get_document_path()
    q_path = _get_questions_path()

    fake_llm = _make_fake_llm()

    with (
        patch("app.api.routes.build_retriever", return_value=_make_fake_retriever()),
        patch("app.services.qa_service._get_llm", return_value=fake_llm),
        open(doc_path, "rb") as doc_f,
        open(q_path, "rb") as q_f,
    ):
        response = client.post(
            "/api/v1/qa",
            files={
                "document_file": (doc_path.name, doc_f, _content_type(doc_path)),
                "questions_file": (q_path.name, q_f, "application/json"),
            },
        )
    return response


class TestQAEndpoint:
    def test_status_200(self, qa_response):
        # A valid document and questions file returns HTTP 200
        assert qa_response.status_code == 200, qa_response.text

    def test_response_is_json(self, qa_response):
        # Response body contains an "answers" key
        data = qa_response.json()
        assert "answers" in data

    def test_all_questions_answered(self, qa_response):
        # Every question in the input file has a corresponding key in the answers dict
        q_path = _get_questions_path()
        questions = json.loads(q_path.read_bytes())
        answers = qa_response.json()["answers"]
        assert set(answers.keys()) == set(questions)

    def test_answers_have_required_fields(self, qa_response):
        # Each answer value is a StructuredAnswer object with the expected fields
        answers = qa_response.json()["answers"]
        for q, a in answers.items():
            assert isinstance(a, dict), f"Answer for '{q}' is not an object"
            assert "answer" in a, f"Answer for '{q}' missing 'answer' field"
            assert "confidence" in a, f"Answer for '{q}' missing 'confidence' field"
            assert "citations" in a, f"Answer for '{q}' missing 'citations' field"
            assert "stepwise_reasoning" in a, f"Answer for '{q}' missing 'stepwise_reasoning' field"
            assert len(a["answer"]) > 0, f"Answer text for '{q}' is empty"

    def test_answers_not_empty(self, qa_response):
        # At least one answer is a real response, not the fallback "Data Not Available"
        answers = qa_response.json()["answers"]
        non_empty = [a for a in answers.values() if a["answer"] != "Data Not Available"]
        assert len(non_empty) > 0, "All answers were 'Data Not Available' — check fixture data or mocks"


class TestQAEndpointValidation:
    def test_unsupported_document_type(self, client):
        # Uploading a non-PDF/JSON document returns 415 with an appropriate error message
        response = client.post(
            "/api/v1/qa",
            files={
                "document_file": ("doc.txt", b"some text", "text/plain"),
                "questions_file": ("q.json", b'["Q?"]', "application/json"),
            },
        )
        assert response.status_code == 415
        assert "Unsupported document type" in response.json()["detail"]["message"]

    def test_invalid_questions_type(self, client):
        # Uploading a non-JSON questions file returns 415 with an appropriate error message
        response = client.post(
            "/api/v1/qa",
            files={
                "document_file": ("doc.json", b'{"k":"v"}', "application/json"),
                "questions_file": ("q.txt", b"questions", "text/plain"),
            },
        )
        assert response.status_code == 415
        assert "Questions file must be JSON" in response.json()["detail"]["message"]

    def test_malformed_questions_json(self, client):
        # A questions file with invalid JSON returns 422
        response = client.post(
            "/api/v1/qa",
            files={
                "document_file": ("doc.json", b'{"k":"v"}', "application/json"),
                "questions_file": ("q.json", b"not json", "application/json"),
            },
        )
        assert response.status_code == 422

    def test_questions_not_array(self, client):
        # A questions file containing a JSON object instead of an array returns 422
        response = client.post(
            "/api/v1/qa",
            files={
                "document_file": ("doc.json", b'{"k":"v"}', "application/json"),
                "questions_file": ("q.json", b'{"q": "not an array"}', "application/json"),
            },
        )
        assert response.status_code == 422

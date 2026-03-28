"""
Integration tests — hit the real OpenAI API.

By default, uses the fixture files in tests/fixtures/.
Override with environment variables:
  TEST_DOCUMENT_PATH  — path to a PDF or JSON document
  TEST_QUESTIONS_PATH — path to a JSON questions file
"""
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

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


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def qa_response(client):
    doc_path = _get_document_path()
    q_path = _get_questions_path()

    with open(doc_path, "rb") as doc_f, open(q_path, "rb") as q_f:
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
        assert qa_response.status_code == 200, qa_response.text

    def test_response_is_json(self, qa_response):
        data = qa_response.json()
        assert "answers" in data

    def test_all_questions_answered(self, qa_response):
        q_path = _get_questions_path()
        questions = json.loads(q_path.read_bytes())
        answers = qa_response.json()["answers"]
        assert set(answers.keys()) == set(questions)

    def test_answers_are_strings(self, qa_response):
        answers = qa_response.json()["answers"]
        for q, a in answers.items():
            assert isinstance(a, str), f"Answer for '{q}' is not a string"
            assert len(a) > 0, f"Answer for '{q}' is empty"

    def test_answers_not_empty(self, qa_response):
        answers = qa_response.json()["answers"]
        # At least one question should have a real answer (not "Data Not Available")
        # since our sample doc covers the sample questions
        non_empty = [a for a in answers.values() if a != "Data Not Available"]
        assert len(non_empty) > 0, "All answers were 'Data Not Available' — check fixture data"


class TestQAEndpointValidation:
    def test_unsupported_document_type(self, client):
        response = client.post(
            "/api/v1/qa",
            files={
                "document_file": ("doc.txt", b"some text", "text/plain"),
                "questions_file": ("q.json", b'["Q?"]', "application/json"),
            },
        )
        assert response.status_code == 415
        assert "Unsupported document type" in response.json()["detail"]

    def test_invalid_questions_type(self, client):
        response = client.post(
            "/api/v1/qa",
            files={
                "document_file": ("doc.json", b'{"k":"v"}', "application/json"),
                "questions_file": ("q.txt", b"questions", "text/plain"),
            },
        )
        assert response.status_code == 415
        assert "Questions file must be JSON" in response.json()["detail"]

    def test_malformed_questions_json(self, client):
        response = client.post(
            "/api/v1/qa",
            files={
                "document_file": ("doc.json", b'{"k":"v"}', "application/json"),
                "questions_file": ("q.json", b"not json", "application/json"),
            },
        )
        assert response.status_code == 422

    def test_questions_not_array(self, client):
        response = client.post(
            "/api/v1/qa",
            files={
                "document_file": ("doc.json", b'{"k":"v"}', "application/json"),
                "questions_file": ("q.json", b'{"q": "not an array"}', "application/json"),
            },
        )
        assert response.status_code == 422

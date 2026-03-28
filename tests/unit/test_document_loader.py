import json

import pytest

from app.services.document_loader import _json_to_chunks, load_documents, load_questions


class TestLoadQuestions:
    def test_valid_questions(self):
        content = json.dumps(["What is X?", "Who owns Y?"]).encode()
        result = load_questions(content, "application/json")
        assert result == ["What is X?", "Who owns Y?"]

    def test_wrong_content_type(self):
        with pytest.raises(ValueError, match="Questions file must be JSON"):
            load_questions(b"data", "application/pdf")

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_questions(b"{not valid json", "application/json")

    def test_not_a_list(self):
        content = json.dumps({"q": "question"}).encode()
        with pytest.raises(ValueError, match="must contain a JSON array"):
            load_questions(content, "application/json")

    def test_non_string_item(self):
        content = json.dumps(["valid", 42]).encode()
        with pytest.raises(ValueError, match="Each question must be a string"):
            load_questions(content, "application/json")

    def test_empty_list(self):
        content = json.dumps([]).encode()
        with pytest.raises(ValueError, match="at least one question"):
            load_questions(content, "application/json")


class TestLoadDocumentsJson:
    def test_valid_json_document(self):
        doc = {"key": "value", "nested": {"a": 1}}
        content = json.dumps(doc).encode()
        docs = load_documents(content, "application/json")
        assert len(docs) > 0
        page_contents = " ".join(d.page_content for d in docs)
        assert "key: value" in page_contents

    def test_empty_json_raises(self):
        content = json.dumps({}).encode()
        with pytest.raises(ValueError, match="empty"):
            load_documents(content, "application/json")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_documents(b"not json", "application/json")

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported document type"):
            load_documents(b"data", "text/plain")


class TestJsonToChunks:
    def test_flat_dict(self):
        chunks = _json_to_chunks({"name": "Alice", "age": 30})
        assert "name: Alice" in chunks
        assert "age: 30" in chunks

    def test_nested_dict(self):
        chunks = _json_to_chunks({"person": {"name": "Bob"}})
        assert any("person.name: Bob" in c for c in chunks)

    def test_list_of_strings(self):
        chunks = _json_to_chunks(["a", "b", "c"])
        assert "[0]: a" in chunks
        assert "[1]: b" in chunks

    def test_nested_list_in_dict(self):
        chunks = _json_to_chunks({"providers": ["AWS", "GCP"]})
        assert any("providers[0]: AWS" in c for c in chunks)
        assert any("providers[1]: GCP" in c for c in chunks)

    def test_scalar(self):
        chunks = _json_to_chunks("hello", prefix="greeting")
        assert chunks == ["greeting: hello"]

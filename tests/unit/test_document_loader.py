import json

import pytest

from app.services.document_loader import _json_to_chunks, load_documents, load_questions


class TestLoadQuestions:
    def test_valid_questions(self):
        # Parses a valid JSON array of strings and returns them unchanged
        content = json.dumps(["What is X?", "Who owns Y?"]).encode()
        result = load_questions(content, "application/json")
        assert result == ["What is X?", "Who owns Y?"]

    def test_wrong_content_type(self):
        # Rejects non-JSON content types with a descriptive error
        with pytest.raises(ValueError, match="Questions file must be JSON"):
            load_questions(b"data", "application/pdf")

    def test_invalid_json(self):
        # Rejects malformed JSON bytes
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_questions(b"{not valid json", "application/json")

    def test_not_a_list(self):
        # Rejects JSON that is valid but not an array
        content = json.dumps({"q": "question"}).encode()
        with pytest.raises(ValueError, match="must contain a JSON array"):
            load_questions(content, "application/json")

    def test_non_string_item(self):
        # Rejects arrays that contain non-string elements
        content = json.dumps(["valid", 42]).encode()
        with pytest.raises(ValueError, match="Each question must be a string"):
            load_questions(content, "application/json")

    def test_empty_list(self):
        # Rejects an empty array since at least one question is required
        content = json.dumps([]).encode()
        with pytest.raises(ValueError, match="at least one question"):
            load_questions(content, "application/json")


class TestLoadDocumentsJson:
    def test_valid_json_document(self):
        # Produces at least one Document and includes flattened key-value text
        doc = {"key": "value", "nested": {"a": 1}}
        content = json.dumps(doc).encode()
        docs = load_documents(content, "application/json")
        assert len(docs) > 0
        page_contents = " ".join(d.page_content for d in docs)
        assert "key: value" in page_contents

    def test_empty_json_raises(self):
        # Rejects an empty JSON object since it yields no chunks
        content = json.dumps({}).encode()
        with pytest.raises(ValueError, match="empty"):
            load_documents(content, "application/json")

    def test_invalid_json_raises(self):
        # Rejects bytes that are not valid JSON
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_documents(b"not json", "application/json")

    def test_unsupported_type_raises(self):
        # Rejects content types other than PDF and JSON
        with pytest.raises(ValueError, match="Unsupported document type"):
            load_documents(b"data", "text/plain")


class TestJsonToChunks:
    def test_flat_dict(self):
        # Flat dict keys become labels with their values as strings
        chunks = _json_to_chunks({"name": "Alice", "age": 30})
        assert ("name", "Alice") in chunks
        assert ("age", "30") in chunks

    def test_nested_dict(self):
        # Nested dict keys are joined with dots to form the label
        chunks = _json_to_chunks({"person": {"name": "Bob"}})
        assert any(label == "person.name" and text == "Bob" for label, text in chunks)

    def test_list_of_strings(self):
        # List items use bracket-index notation as their label
        chunks = _json_to_chunks(["a", "b", "c"])
        assert ("[0]", "a") in chunks
        assert ("[1]", "b") in chunks

    def test_nested_list_in_dict(self):
        # List nested inside a dict combines the key and bracket-index
        chunks = _json_to_chunks({"providers": ["AWS", "GCP"]})
        assert any(label == "providers[0]" and text == "AWS" for label, text in chunks)
        assert any(label == "providers[1]" and text == "GCP" for label, text in chunks)

    def test_scalar(self):
        # A bare scalar uses the given prefix as its label
        chunks = _json_to_chunks("hello", prefix="greeting")
        assert chunks == [("greeting", "hello")]

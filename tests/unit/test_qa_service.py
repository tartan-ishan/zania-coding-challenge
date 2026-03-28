from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.services.qa_service import answer_questions, _answer_single


def _make_retriever(docs: list[Document]):
    """Return a mock retriever that always returns the given docs."""
    retriever = MagicMock()
    retriever.ainvoke = AsyncMock(return_value=docs)
    return retriever


HIGH_SCORE_DOC = Document(page_content="AWS and GCP are used as cloud providers.")
IRRELEVANT_DOC = Document(page_content="x")  # short/noisy — still returned by retriever


class TestAnswerSingle:
    @pytest.mark.asyncio
    async def test_returns_data_not_available_when_no_chunks(self):
        retriever = _make_retriever([])
        with patch("app.services.qa_service._decompose_question", new=AsyncMock(return_value=["Q?"])):
            result = await _answer_single("What is X?", retriever)
        assert result == "Data Not Available"

    @pytest.mark.asyncio
    async def test_calls_llm_when_chunks_retrieved(self):
        retriever = _make_retriever([HIGH_SCORE_DOC])
        with (
            patch("app.services.qa_service._decompose_question", new=AsyncMock(return_value=["Q?"])),
            patch("app.services.qa_service._call_llm", new=AsyncMock(return_value="AWS and GCP")),
        ):
            result = await _answer_single("Which cloud providers?", retriever)
        assert result == "AWS and GCP"

    @pytest.mark.asyncio
    async def test_deduplicates_chunks_across_sub_queries(self):
        duplicate_doc = Document(page_content="same content")
        retriever = MagicMock()
        # Both sub-queries return the same doc
        retriever.ainvoke = AsyncMock(return_value=[duplicate_doc])

        with (
            patch("app.services.qa_service._decompose_question", new=AsyncMock(return_value=["Q1?", "Q2?"])),
            patch("app.services.qa_service._call_llm", new=AsyncMock(return_value="answer")) as mock_llm,
        ):
            await _answer_single("Question?", retriever)

        # Context passed to LLM should contain the doc only once
        context_arg = mock_llm.call_args.kwargs["context"]
        assert context_arg.count("same content") == 1


class TestAnswerQuestions:
    @pytest.mark.asyncio
    async def test_returns_dict_keyed_by_question(self):
        retriever = _make_retriever([HIGH_SCORE_DOC])
        questions = ["Q1?", "Q2?"]
        with (
            patch("app.services.qa_service._decompose_question", new=AsyncMock(return_value=["Q?"])),
            patch("app.services.qa_service._call_llm", new=AsyncMock(return_value="some answer")),
        ):
            result = await answer_questions(questions, retriever)
        assert set(result.keys()) == {"Q1?", "Q2?"}

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        retriever = _make_retriever([HIGH_SCORE_DOC])
        questions = [f"Question {i}?" for i in range(5)]
        with (
            patch("app.services.qa_service._decompose_question", new=AsyncMock(return_value=["Q?"])),
            patch("app.services.qa_service._call_llm", new=AsyncMock(return_value="answer")),
        ):
            result = await answer_questions(questions, retriever)
        assert len(result) == 5


class TestDecomposeQuestion:
    @pytest.mark.asyncio
    async def test_falls_back_to_original_on_failure(self):
        from app.services.qa_service import _decompose_question

        with patch("app.services.qa_service._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
            mock_llm_fn.return_value = mock_llm

            result = await _decompose_question("Original question?", count=3)

        assert "Original question?" in result

    @pytest.mark.asyncio
    async def test_always_includes_original_question(self):
        from app.services.qa_service import _decompose_question

        with patch("app.services.qa_service._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(
                return_value=MagicMock(content="sub-query 1\nsub-query 2\nsub-query 3")
            )
            mock_llm_fn.return_value = mock_llm

            result = await _decompose_question("Original question?", count=3)

        assert "Original question?" in result

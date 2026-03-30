from unittest.mock import MagicMock, patch

import pytest
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.services.vector_store import build_retriever


def _real_bm25_retriever(docs):
    """Build a real BM25Retriever (no network calls needed)."""
    r = BM25Retriever.from_documents(docs)
    r.k = 5
    return r


class TestBuildRetriever:
    def test_returns_ensemble_retriever(self):
        # build_retriever returns an EnsembleRetriever combining semantic and keyword search
        docs = [
            Document(page_content="AWS is the primary cloud provider."),
            Document(page_content="Data is encrypted using AES-256."),
            Document(page_content="Incident response is handled by the CISO."),
        ]
        mock_semantic = _real_bm25_retriever(docs)  # a real Runnable, used as stand-in

        with patch("app.services.vector_store.Chroma.from_documents") as mock_chroma_cls:
            mock_chroma_cls.return_value.as_retriever.return_value = mock_semantic
            retriever = build_retriever(docs)

        assert isinstance(retriever, EnsembleRetriever)

    def test_ensemble_has_two_retrievers(self):
        # The ensemble wraps exactly two sub-retrievers (semantic + BM25)
        docs = [
            Document(page_content="AWS is used for hosting."),
            Document(page_content="GCP is used for analytics."),
        ]
        mock_semantic = _real_bm25_retriever(docs)

        with patch("app.services.vector_store.Chroma.from_documents") as mock_chroma_cls:
            mock_chroma_cls.return_value.as_retriever.return_value = mock_semantic
            retriever = build_retriever(docs)

        assert len(retriever.retrievers) == 2

    def test_weights_sum_to_one(self):
        # Retriever weights are normalised so they sum to exactly 1.0
        docs = [Document(page_content="Some content about security policies.")]
        mock_semantic = _real_bm25_retriever(docs)

        with patch("app.services.vector_store.Chroma.from_documents") as mock_chroma_cls:
            mock_chroma_cls.return_value.as_retriever.return_value = mock_semantic
            retriever = build_retriever(docs)

        assert abs(sum(retriever.weights) - 1.0) < 1e-6

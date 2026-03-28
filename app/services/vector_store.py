from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_classic.retrievers import EnsembleRetriever

from app.config import get_settings


def build_retriever(documents: list[Document]) -> BaseRetriever:
    """
    Build a hybrid retriever combining:
      - Chroma vector store with MMR (semantic, diverse)
      - BM25 (keyword / lexical)

    Results from both are fused via Reciprocal Rank Fusion (EnsembleRetriever).

    To switch to persistent Chroma: pass persist_directory to Chroma().
    To switch to Pinecone: replace the Chroma retriever with a Pinecone-backed one.
    The EnsembleRetriever interface stays the same.
    """
    settings = get_settings()

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    # Semantic retriever with MMR for diversity
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    semantic_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retrieval_k,
            "fetch_k": settings.mmr_fetch_k,
            "lambda_mult": settings.mmr_lambda,
        },
    )

    # Keyword retriever (BM25 — no embedding calls)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = settings.retrieval_k

    # Fuse results via Reciprocal Rank Fusion
    semantic_weight = round(1.0 - settings.bm25_weight, 4)
    return EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[semantic_weight, settings.bm25_weight],
    )

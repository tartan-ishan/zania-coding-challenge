import asyncio
import logging

import openai
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# _DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are a search query expert for compliance and security documents (SOC 2, ISO 27001, vendor assessments).\n\n"
#         "Generate exactly {count} search queries for the given question using this mix:\n"
#         "  1. One plain restatement of the question in simple natural language.\n"
#         "  2. One or two queries using compliance document vocabulary "
#         "(e.g. 'third parties' → 'subservice organizations'; "
#         "'personal information' → 'PII Customer Confidential data'; "
#         "'cloud providers' → 'hosting infrastructure subservice organizations'; "
#         "'monitoring' → 'availability monitoring utilization metrics audit events'; "
#         "'incident notification SLA' → 'data breach notification policy incident severity escalation').\n"
#         "  3. One short keyword phrase (4-6 words max) targeting the most specific fact needed.\n"
#         "  4. One control-framework query if applicable (e.g. 'CC7.3 incident response notification', 'A1.1 availability monitoring capacity').\n\n"
#         "Output ONLY the queries, one per line, no numbering, no explanation.",
#     ),
#     ("human", "{question}"),
# ])

_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert at retrieving information from compliance, security, and regulatory documents "
        "(e.g. SOC 2, ISO 27001, GDPR, HIPAA, PCI DSS, NIST, vendor risk assessments, penetration test reports, "
        "privacy policies, data processing agreements).\n\n"
        "Your task: given a question, generate {count} distinct search queries that together maximise "
        "the chance of finding the answer across differently-worded compliance documents.\n\n"
        "Apply these strategies — use whichever are relevant, in any order:\n\n"
        "  PLAIN RESTATEMENT — rephrase the question in simple, direct natural language.\n\n"
        "  FORMAL/DOCUMENT VOCABULARY — replace everyday terms with the language compliance documents "
        "typically use. Examples of substitutions (apply the principle, don't copy literally):\n"
        "    · personal data → PII / customer confidential data / data subjects\n"
        "    · third parties → subservice organizations / vendors / processors / subcontractors\n"
        "    · cloud providers → hosting infrastructure / subservice organizations / IaaS PaaS providers\n"
        "    · monitoring → availability monitoring / audit logging / utilization metrics / alerting\n"
        "    · breach notification → incident escalation / data breach notification SLA / severity classification\n"
        "    · access control → logical access / privileged access management / least privilege\n"
        "    · employees → workforce / personnel / authorized users\n\n"
        "  ACRONYM / SYNONYM EXPANSION — include alternate abbreviations or synonyms the document may use "
        "(e.g. MFA / 2FA / multi-factor authentication; DR / BCP / business continuity; "
        "VAPT / penetration testing / vulnerability assessment).\n\n"
        "  SPECIFIC FACT TARGETING — a short keyword phrase (3-6 words) aimed at the precise data point, "
        "metric, or clause being asked about (e.g. 'RTO RPO recovery objectives', "
        "'encryption at rest AES-256', 'background check pre-employment screening').\n\n"
        "  CONTROL FRAMEWORK ANCHORING — if the question maps to a known control objective, include a "
        "query referencing the relevant framework language "
        "(e.g. 'CC6.1 logical access controls', 'A1.2 environmental protections', "
        "'ISO 27001 A.12.3 backup', 'NIST CSF PR.AC identity management').\n\n"
        "Rules:\n"
        "  - Output ONLY the queries, one per line, no numbering, no labels, no explanation.\n"
        "  - Each query must be meaningfully distinct — no paraphrasing the same query twice.\n"
        "  - Do not hallucinate control numbers; only include framework references you are confident apply.\n"
        "  - Prefer breadth over depth: cover different angles rather than slight rewording.",
    ),
    ("human", "{question}"),
])

# _KEYWORD_EXPAND_PROMPT = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are a compliance document search expert.\n\n"
#         "Given a question, output a flat JSON array of 5-8 short keyword strings "
#         "that a SOC 2 or security compliance document would use when discussing the answer. "
#         "Focus on proper nouns, acronyms, policy names, technical terms, and control IDs "
#         "that would appear verbatim in the document.\n\n"
#         "Examples for 'personal information third parties': "
#         '[\"PII\", \"Customer Confidential\", \"subservice organization\", \"data classification\", \"non-disclosure agreement\", \"vendor risk assessment\"]\n\n'
#         "Output ONLY the JSON array, no explanation.",
#     ),
#     ("human", "{question}"),
# ])

_KEYWORD_EXPAND_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert at keyword extraction for compliance and security document retrieval.\n\n"
        "Given a question, output a flat JSON array of short keyword strings optimised for BM25 lexical search "
        "against compliance documents (SOC 2, ISO 27001, GDPR, HIPAA, PCI DSS, NIST, vendor assessments, "
        "penetration test reports, data processing agreements).\n\n"
        "Include keywords from TWO categories:\n\n"
        "  1. ANCHOR TERMS — high-signal words and phrases taken directly or near-directly from the question "
        "that must not be lost during retrieval. These are the terms whose absence would cause a relevant "
        "document to be missed. Preserve them even if they seem obvious.\n\n"
        "  2. DOCUMENT VOCABULARY EXPANSIONS — alternative terms, acronyms, synonyms, and formal compliance "
        "language that documents use when discussing the same concept. Focus on:\n"
        "    · Proper nouns and acronyms (e.g. PII, MFA, VAPT, BCP, RTO, DPA)\n"
        "    · Policy and control names (e.g. 'access control policy', 'incident response plan')\n"
        "    · Framework control IDs only when you are confident they apply "
        "(e.g. 'CC6.1', 'A.9.2', 'PR.AC')\n"
        "    · Document-register terms (e.g. 'subservice organization', 'data classification', "
        "'vendor risk assessment', 'statement of applicability')\n\n"
        "Rules:\n"
        "  - Output 6-10 strings total, prioritising terms most likely to appear verbatim in a compliance document.\n"
        "  - Keep each string short (1-4 words max) — these are keywords, not sentences.\n"
        "  - No duplicates or near-duplicates.\n"
        "  - Do not invent control IDs or policy names you are not confident exist.\n"
        "  - Output ONLY the JSON array, no explanation, no markdown fences.",
    ),
    ("human", "{question}"),
])

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise question-answering assistant working with compliance and security documents "
        "such as SOC 2 reports. Follow these rules strictly:\n\n"

        "STRUCTURE:\n"
        "- Open with a direct statement answering the question (e.g. 'Yes, Product Fruits has...' "
        "or 'The sources do not explicitly use the term X, however they describe...'). "
        "Never open with 'The context specifies' or 'The provided context'.\n"
        "- For multi-part questions, use a bolded sub-header per part "
        "(e.g. '**Notification Criteria:**', '**Transmission:**', '**Storage and Retention:**').\n"
        "- Synthesise across ALL context passages — do not stop at the first relevant fact.\n\n"

        "TERMINOLOGY BRIDGING:\n"
        "- When the document uses different terms than the question, bridge them explicitly "
        "(e.g. 'third parties (referred to as subservice organizations in this document)', "
        "'monitoring (described as utilization metrics and audit event generation)').\n\n"

        "PARTIAL INFORMATION:\n"
        "- For each part of the question, use the pattern: state what IS documented, "
        "then note what is NOT specified — in that order.\n"
        "- Never append 'Data Not Available' as a trailing sentence after providing partial content. "
        "Instead write: 'The sources do not specify [specific missing detail].'\n"
        "- Only respond with exactly 'Data Not Available' (and nothing else) when the context "
        "contains zero relevant information for the entire question.\n\n"

        "ACCURACY:\n"
        "- Do not use outside knowledge. Do not speculate beyond what the context states.\n\n"

        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def answer_questions(
    questions: list[str],
    retriever: BaseRetriever,
) -> dict[str, str]:
    """Answer all questions concurrently against the hybrid retriever."""
    tasks = [_answer_single(q, retriever) for q in questions]
    results = await asyncio.gather(*tasks)
    return dict(zip(questions, results))


# ---------------------------------------------------------------------------
# Internal pipeline
# ---------------------------------------------------------------------------

async def _answer_single(question: str, retriever: BaseRetriever) -> str:
    settings = get_settings()

    # Step 1: decompose + keyword expand concurrently
    sub_queries, keywords = await asyncio.gather(
        _decompose_question(question, settings.multi_query_count),
        _expand_keywords(question),
    )
    logger.info("Sub-queries for %r: %s", question[:60], sub_queries)
    logger.info("Keywords for %r: %s", question[:60], keywords)

    # Step 2: retrieve for sub-queries first (ordered by specificity), then keywords.
    # Sub-queries are retrieved first so their results lead the context — keywords are additive.
    all_queries = sub_queries + keywords
    retrieve_tasks = [retriever.ainvoke(q) for q in all_queries]
    all_results = await asyncio.gather(*retrieve_tasks)

    all_docs: list[Document] = []
    seen: set[str] = set()
    for docs in all_results:
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

    if not all_docs:
        logger.info("No chunks retrieved for question: %r", question[:60])
        return "Data Not Available"

    max_chunks = settings.retrieval_k * 3
    context = "\n\n".join(doc.page_content for doc in all_docs[:max_chunks])
    return await _call_llm(question=question, context=context)


async def _decompose_question(question: str, count: int) -> list[str]:
    """Use the LLM to generate `count` balanced sub-queries from the original question."""
    try:
        response = await _get_llm().ainvoke(
            _DECOMPOSE_PROMPT.format_messages(question=question, count=count)
        )
        lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
        sub_queries = lines[:count] if lines else [question]
    except Exception as e:
        logger.warning("Query decomposition failed (%s), falling back to original", e)
        sub_queries = [question]

    # Always include the original question to preserve simple semantic signal
    if question not in sub_queries:
        sub_queries.insert(0, question)

    return sub_queries


async def _expand_keywords(question: str) -> list[str]:
    """
    Use the LLM to generate document-vocabulary keywords for BM25 retrieval.
    Returns short keyword strings that would appear verbatim in a compliance doc.
    Falls back to empty list on failure — keywords are additive, not required.
    """
    import json as _json
    try:
        response = await _get_llm().ainvoke(
            _KEYWORD_EXPAND_PROMPT.format_messages(question=question)
        )
        raw = response.content.strip()
        keywords = _json.loads(raw)
        if isinstance(keywords, list):
            return [str(k).strip() for k in keywords if k]
    except Exception as e:
        logger.warning("Keyword expansion failed (%s), skipping", e)
    return []


@retry(
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIStatusError)
    ),
    stop=stop_after_attempt(get_settings().max_retries),
    wait=wait_exponential(
        min=get_settings().retry_min_wait,
        max=get_settings().retry_max_wait,
    ),
    reraise=True,
)
async def _call_llm(question: str, context: str) -> str:
    try:
        response = await _get_llm().ainvoke(
            _ANSWER_PROMPT.format_messages(question=question, context=context)
        )
        return response.content.strip()
    except openai.RateLimitError:
        logger.warning("OpenAI rate limit hit, retrying...")
        raise
    except openai.APIStatusError as e:
        logger.error("OpenAI API error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected LLM error: %s", e)
        raise RuntimeError(f"LLM call failed: {e}") from e


def _get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

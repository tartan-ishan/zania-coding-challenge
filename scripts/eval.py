"""
LLM-as-judge evaluation script against a golden dataset.

For each question the judge scores the system answer on three axes (1-10):
  - completeness : does it cover all parts the ideal answer covers?
  - accuracy     : is everything stated factually correct per the ideal?
  - phrasing     : does it handle partial information gracefully
                   (e.g. noting what's missing rather than refusing)?

Usage:
    uv run python scripts/eval.py
    uv run python scripts/eval.py --document sample_docs/soc2-type2.pdf \\
                                   --golden   sample_docs/golden_dataset.json

Results are printed to stdout and saved to sample_docs/eval_results.json.
Exit code: 0 if average score across all axes >= 7, else 1.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_classic.evaluation import EvaluatorType, load_evaluator
from langchain_openai import ChatOpenAI

from app.services.document_loader import load_documents
from app.services.qa_service import answer_questions
from app.services.vector_store import build_retriever

PASS_SCORE = 5        # minimum average score (out of 10) to pass a question
OVERALL_PASS = 6      # minimum overall average to exit 0

CRITERIA = {
    "completeness": (
        "Score whether the prediction covers the key factual points from the reference answer. "
        "Focus ONLY on whether the core facts are present — do NOT penalise for being shorter, "
        "less detailed, or differently structured than the reference as long as the key information is there. "
        "You MUST assign a score from the bands below — no other values are valid:\n"
        "  1-2 — Misses almost all key facts from the reference; only a trivial fragment is present.\n"
        "  3-4 — Captures a minority of the key facts; most important information is absent.\n"
        "  5-6 — Captures roughly half the key facts; some important points covered, others missing.\n"
        "  7-8 — Captures most key facts; only minor or peripheral details absent.\n"
        "  9-10 — All key facts from the reference are present, even if expressed more concisely.\n"
        "Within each band, use the lower number if the prediction is at the weaker end of that description, "
        "and the higher number if it is at the stronger end. Output only the integer."
    ),
    "accuracy": (
        "Score the factual correctness of every claim in the prediction against the reference answer. "
        "You MUST assign a score from the bands below — no other values are valid:\n"
        "  1-2 — Multiple significant factual errors or hallucinations not supported by the reference.\n"
        "  3-4 — At least one clear factual error or overconfident claim unsupported by the reference.\n"
        "  5-6 — Mostly correct but includes a minor inaccuracy or unsupported generalisation.\n"
        "  7-8 — All claims accurate; at most one very minor imprecision.\n"
        "  9-10 — Every factual claim fully supported by and consistent with the reference.\n"
        "Within each band, use the lower number if the prediction is at the weaker end of that description, "
        "and the higher number if it is at the stronger end. Output only the integer."
    ),
    "phrasing": (
        "Score how well the prediction handles partial information — stating what IS documented "
        "and explicitly noting what is NOT specified, rather than refusing entirely. "
        "You MUST assign a score from the bands below — no other values are valid:\n"
        "  1-2 — Refuses to answer ('Data Not Available') despite partial information existing, "
        "OR answers with no caveats when key details are clearly missing.\n"
        "  3-4 — Provides some content but appends a blanket 'Data Not Available' that contradicts "
        "the partial content, or omits important per-topic caveats.\n"
        "  5-6 — Acknowledges missing information but the caveat is vague or only stated once at the end "
        "rather than per sub-topic.\n"
        "  7-8 — Clearly states what is documented and notes per-topic what is not specified; "
        "minor phrasing awkwardness only.\n"
        "  9-10 — Direct opening, per-topic coverage, explicit statements of what sources do and do not "
        "specify, no contradictory trailing 'Data Not Available'.\n"
        "Within each band, use the lower number if the prediction is at the weaker end of that description, "
        "and the higher number if it is at the stronger end. Output only the integer."
    ),
}


async def _judge_answer(
    evaluator_map: dict,
    question: str,
    system_answer: str,
    ideal_answer: str,
) -> dict[str, dict]:
    """Run all criteria evaluators concurrently for a single Q/A pair."""
    tasks = {
        criterion: evaluator_map[criterion].aevaluate_strings(
            prediction=system_answer,
            reference=ideal_answer,
            input=question,
        )
        for criterion in CRITERIA
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    return {
        criterion: (
            result if not isinstance(result, Exception)
            else {"score": None, "reasoning": str(result)}
        )
        for criterion, result in zip(tasks.keys(), results)
    }


async def run_eval(document_path: Path, golden_path: Path) -> int:
    print(f"Document : {document_path}")
    print(f"Golden   : {golden_path}")
    print(f"Pass score: {PASS_SCORE}/10 per criterion\n")

    with open(document_path, "rb") as f:
        doc_bytes = f.read()
    content_type = "application/pdf" if document_path.suffix == ".pdf" else "application/json"

    with open(golden_path) as f:
        golden = json.load(f)

    # Skip entries without an ideal answer
    entries = [e for e in golden if e.get("ideal_answer", "").strip()]
    if not entries:
        print("No entries with ideal answers found in golden dataset.")
        return 1

    questions = [e["question"] for e in entries]

    # Build retriever once
    print("Building retriever (embedding document)...")
    docs = load_documents(doc_bytes, content_type)
    retriever = build_retriever(docs)

    print(f"Answering {len(questions)} questions...\n")
    answers = await answer_questions(questions, retriever)

    # Build one evaluator per criterion (each uses a separate prompt)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_map = {
        criterion: load_evaluator(
            EvaluatorType.LABELED_SCORE_STRING,
            llm=llm,
            criteria=description,
        )
        for criterion, description in CRITERIA.items()
    }

    print("Judging answers...\n")
    all_results = []
    all_scores: list[float] = []

    for entry in entries:
        q = entry["question"]
        ideal = entry["ideal_answer"]
        system = answers[q]

        scores_for_q = await _judge_answer(evaluator_map, q, system, ideal)

        numeric_scores = [
            v["score"] for v in scores_for_q.values() if v.get("score") is not None
        ]
        avg = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0
        all_scores.extend(numeric_scores)
        passed = avg >= PASS_SCORE

        result = {
            "question": q,
            "system_answer": system,
            "ideal_answer": ideal,
            "scores": {
                criterion: {
                    "score": v.get("score"),
                    "reasoning": v.get("reasoning", ""),
                }
                for criterion, v in scores_for_q.items()
            },
            "average_score": round(avg, 2),
            "pass": passed,
        }
        all_results.append(result)

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}  avg={avg:.1f}/10  |  {q[:80]}")
        for criterion, v in scores_for_q.items():
            score_str = str(v.get("score", "err"))
            reasoning = v.get("reasoning", "")
            # Print first sentence of reasoning only
            first_sentence = reasoning.split(".")[0].strip() if reasoning else ""
            print(f"    {criterion:14s} {score_str:>3}/10  {first_sentence[:80]}")
        print(f"  System: {system[:180]}...")
        print()

    # Save results
    out_path = golden_path.parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}\n")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    passed_count = sum(1 for r in all_results if r["pass"])
    print(f"Summary: {passed_count}/{len(all_results)} questions passed  |  overall avg={overall:.2f}/10")

    return 0 if overall >= OVERALL_PASS else 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate QA pipeline against a golden dataset")
    parser.add_argument(
        "--document",
        type=Path,
        default=Path("sample_docs/soc2-type2.pdf"),
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("sample_docs/golden_dataset.json"),
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run_eval(args.document, args.golden)))


if __name__ == "__main__":
    main()

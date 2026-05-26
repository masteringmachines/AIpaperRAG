"""
Evaluate RAG quality using RAGAS metrics.
"""

from __future__ import annotations

from typing import List, Dict, Any


def build_test_dataset(qa_pairs: List[Dict[str, Any]]):
    """Convert QA pairs into a HuggingFace Dataset for RAGAS.

    Parameters
    ----------
    qa_pairs:
        List of dicts with keys: ``question``, ``answer``, ``contexts``.
        ``contexts`` is a list of strings (retrieved chunk texts).

    Example
    -------
    >>> qa_pairs = [
    ...     {
    ...         "question": "What is chain-of-thought?",
    ...         "answer": "CoT prompting breaks down reasoning into steps...",
    ...         "contexts": ["Chain-of-thought enables step-by-step reasoning..."],
    ...     }
    ... ]
    >>> dataset = build_test_dataset(qa_pairs)
    """
    from datasets import Dataset

    data = {
        "question": [p["question"] for p in qa_pairs],
        "answer": [p["answer"] for p in qa_pairs],
        "contexts": [p["contexts"] for p in qa_pairs],
    }
    return Dataset.from_dict(data)


def run_ragas_evaluation(dataset, metrics=None):
    """Run RAGAS evaluation on a dataset.

    Parameters
    ----------
    dataset:
        HuggingFace Dataset with ``question``, ``answer``, ``contexts`` columns.
    metrics:
        List of RAGAS metric objects.  Defaults to faithfulness + answer_relevancy + context_recall.

    Returns
    -------
    dict with metric scores.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall

    if metrics is None:
        metrics = [faithfulness, answer_relevancy, context_recall]

    print("[evaluator] Running RAGAS evaluation …")
    result = evaluate(dataset, metrics=metrics)
    print("[evaluator] Results:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
    return result


def evaluate_from_chain(chain, retriever, qa_pairs: List[Dict[str, str]]):
    """End-to-end evaluation helper.

    For each QA pair, runs the chain and retriever to produce answers and contexts,
    then evaluates with RAGAS.

    Parameters
    ----------
    chain:
        Built RAG chain.
    retriever:
        The retriever used in the chain (to collect contexts separately).
    qa_pairs:
        List of dicts with ``question`` and optionally ``ground_truth``.

    Returns
    -------
    RAGAS result dict.
    """
    enriched = []
    for pair in qa_pairs:
        question = pair["question"]
        docs = retriever.invoke(question)
        contexts = [d.page_content for d in docs]
        answer = chain.invoke(question)
        enriched.append(
            {"question": question, "answer": answer, "contexts": contexts}
        )

    dataset = build_test_dataset(enriched)
    return run_ragas_evaluation(dataset)

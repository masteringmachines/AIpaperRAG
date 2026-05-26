"""Unit tests for prompt templates."""

from src.chain.prompts import RAG_PROMPT, FEW_SHOT_RAG_PROMPT


def test_rag_prompt_variables():
    """Prompt should require 'context' and 'question'."""
    variables = RAG_PROMPT.input_variables
    assert "context" in variables
    assert "question" in variables


def test_rag_prompt_renders():
    """Prompt should render without errors."""
    messages = RAG_PROMPT.format_messages(
        context="Some context text.",
        question="What is RAG?",
    )
    assert len(messages) >= 2


def test_few_shot_prompt_renders():
    messages = FEW_SHOT_RAG_PROMPT.format_messages(
        context="Context about LoRA.",
        question="What is LoRA?",
    )
    assert any("LoRA" in str(m) for m in messages)

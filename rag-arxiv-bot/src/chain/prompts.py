"""
Prompt templates for the RAG chain.
"""

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# ---------------------------------------------------------------------------
# Main Q&A prompt
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are an expert AI research assistant. "
    "Answer the user's question based ONLY on the provided context excerpts from "
    "academic papers. "
    "Rules:\n"
    "1. If the context does not contain enough information to answer, say so clearly.\n"
    "2. Always end your answer with a 'Sources:' section listing every paper you drew from, "
    "in the format: (Title · Authors · Year).\n"
    "3. Be concise and precise. Prefer bullet points for multi-part answers.\n"
    "4. Do NOT fabricate information or cite papers not in the context."
)

HUMAN_TEMPLATE = """\
Context excerpts:
{context}

---
Question: {question}

Answer (with citations):"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        ("human", HUMAN_TEMPLATE),
    ]
)

# ---------------------------------------------------------------------------
# Few-shot variant (optional — uncomment to use)
# ---------------------------------------------------------------------------

_EXAMPLES = [
    {
        "question": "What is LoRA?",
        "answer": (
            "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that "
            "inserts trainable low-rank matrices into transformer layers while keeping the "
            "original weights frozen, dramatically reducing the number of trainable parameters.\n\n"
            "Sources:\n"
            "- LoRA: Low-Rank Adaptation of Large Language Models · Hu et al. · 2021"
        ),
    },
    {
        "question": "What is chain-of-thought prompting?",
        "answer": (
            "Chain-of-thought (CoT) prompting elicits step-by-step reasoning from LLMs by "
            "providing examples that show intermediate reasoning steps, improving performance "
            "on arithmetic, commonsense, and symbolic reasoning tasks.\n\n"
            "Sources:\n"
            "- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models · Wei et al. · 2022"
        ),
    },
]

_example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{question}"), ("ai", "{answer}")]
)

FEW_SHOT_PROMPT = FewShotChatMessagePromptTemplate(
    example_prompt=_example_prompt,
    examples=_EXAMPLES,
)

FEW_SHOT_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        FEW_SHOT_PROMPT,
        ("human", HUMAN_TEMPLATE),
    ]
)

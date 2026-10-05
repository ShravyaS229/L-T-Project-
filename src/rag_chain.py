import os

from dotenv import load_dotenv
from openai import OpenAI

from src.prompts import RAG_PROMPT
from src.retriever import search_documents

load_dotenv()

NOT_FOUND_MESSAGE = "I couldn't find this information in the college documents."


def format_context(chunks):
    """Combine retrieved chunks into one text block for the prompt."""
    parts = []
    for chunk in chunks:
        parts.append(
            f"[Source: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def format_chat_history(chat_history):
    """Format previous session messages for reference resolution only."""
    if not chat_history:
        return "No previous conversation."

    messages = []
    for message in chat_history:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).title()
        content = str(message.get("content", "")).strip()
        if content:
            messages.append(f"{role}: {content}")
    return "\n".join(messages) or "No previous conversation."


def retrieve_context(question, k=4):
    """Step 1: get relevant PDF chunks from Person 1's vector DB."""
    chunks = search_documents(question, k=k)
    return chunks, format_context(chunks)


def get_client():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is missing. Copy .env.example to .env and fill it in."
        )
    # LLM_BASE_URL is empty for OpenAI itself, set for Groq/OpenRouter/Gemini
    base_url = os.getenv("LLM_BASE_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url)


def ask_question(question, k=4, chat_history=None):
    """Full RAG pipeline. Returns {"answer": str, "sources": [str, ...]}."""
    chunks, context = retrieve_context(question, k=k)

    if not chunks:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    prompt = RAG_PROMPT.format(
        context=context,
        question=question,
        chat_history=format_chat_history(chat_history),
    )

    client = get_client()
    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_MODEL is missing in your .env file.")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()

    sources = []
    for chunk in chunks:
        label = f"{chunk['source']}, Page {chunk['page']}"
        if label not in sources:
            sources.append(label)

    return {"answer": answer, "sources": sources}

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from config import (
    GROQ_API_KEY, GEMINI_API_KEY,
    GROQ_MODEL, GEMINI_MODEL,
    LLM_PROVIDER, TEMPERATURE, TOP_K
)
from retriever import retrieve, format_context, get_source_references

TROUBLESHOOT_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template="""You are **AutoDiagGPT**, an experienced field technician's assistant for off-highway and commercial vehicles.

**STRICT INSTRUCTIONS:**
- Answer ONLY based on the provided context below.
- If the context does not contain enough information, clearly state: "Insufficient data in the knowledge base for this query."
- Do NOT hallucinate or make up information.
- Be specific, actionable, and safety-conscious.

Retrieved Context:
{context}

Technician's Query: {query}

Give a direct, point-wise answer. Use short bullet points or numbered steps — no long paragraphs, no rigid section headers like "Probable Causes" or "Corrective Actions".
Just get straight to: what's likely wrong, what to check, and how to fix it, in clear bullet points.
Include a safety point only if it's directly relevant to this specific issue — keep it brief and inline with the other points, not as a separate section.
Keep the whole answer concise — aim for 4-8 bullet points total.
"""
)

CONVERSATION_PROMPT = PromptTemplate(
    input_variables=["context", "query", "chat_history"],
    template="""You are **AutoDiagGPT**, an experienced field technician's assistant.

Previous Conversation:
{chat_history}

Retrieved Context:
{context}

Technician's Query: {query}

Answer ONLY based on the context. Give a direct, point-wise answer using short bullet points or numbered steps — no long paragraphs, no rigid section headers.
Be specific, concise, and safety-conscious. If the context doesn't have enough info, say so clearly instead of guessing.
"""
)


def get_llm(provider: str = None):
    provider = provider or LLM_PROVIDER

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=2048,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
            max_output_tokens=2048,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'groq' or 'gemini'.")


def ask(query: str, system_filter: str = None,
        chat_history: str = "", provider: str = None) -> dict:
    # Step 1: RETRIEVE (LOCAL on Jetson)
    results = retrieve(query, system_filter=system_filter, top_k=TOP_K)
    context = format_context(results)
    sources = get_source_references(results)

    # Step 2: GENERATE (CLOUD — Free API)
    used_provider = provider or LLM_PROVIDER
    try:
        llm = get_llm(used_provider)
        if chat_history:

            prompt = CONVERSATION_PROMPT.format(
            context=context,
            query=query,
            chat_history=chat_history
        )

        else:

            prompt = TROUBLESHOOT_PROMPT.format(
            context=context,
            query=query
        )

        response = llm.invoke(prompt)

        answer = response.content

        return {"answer": answer, "sources": sources,
                "num_chunks_retrieved": len(results),
                "provider": used_provider, "status": "success"}
    except Exception as e:
        return {"answer": f"Error from {used_provider}: {str(e)}", "sources": sources,
                "num_chunks_retrieved": len(results),
                "provider": used_provider, "status": "error"}


def ask_with_fallback(query: str, system_filter: str = None,
                      chat_history: str = "") -> dict:
    """Try Groq first, fallback to Gemini on rate limit."""
    providers = ["groq", "gemini"]
    for provider in providers:
        result = ask(query=query, system_filter=system_filter,
                     chat_history=chat_history, provider=provider)
        if result["status"] == "success":
            return result
        error = result["answer"].lower()
        if "rate limit" in error or "429" in error or "quota" in error:
            print(f"Warning: {provider} rate limited, trying next...")
            continue
        else:
            return result

    return {"answer": "All free API providers are rate-limited. Please wait and try again.",
            "sources": [], "num_chunks_retrieved": 0,
            "provider": "none", "status": "error"}


if __name__ == "__main__":
    test_query = "Hydraulic pump not building pressure, error code E45"
    print(f"Query: {test_query}")
    print(f"Provider: {LLM_PROVIDER}\n")
    result = ask_with_fallback(test_query, system_filter="hydraulics")
    print(result["answer"])
    print(f"\nProvider used: {result['provider']}")
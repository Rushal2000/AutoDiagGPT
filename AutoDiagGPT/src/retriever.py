from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import VECTORSTORE_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, TOP_K


def load_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        num_gpu=0    # directly as top-level param
    )
    return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)


def retrieve(query: str, system_filter: str = None, top_k: int = TOP_K) -> list:
    vectorstore = load_vectorstore()
    search_kwargs = {"k": top_k}
    if system_filter and system_filter != "all":
        search_kwargs["filter"] = {"system": system_filter}
    results = vectorstore.similarity_search_with_relevance_scores(query, **search_kwargs)
    return results


def format_context(results: list) -> str:
    if not results:
        return "No relevant documents found in the knowledge base."
    context_parts = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        system = doc.metadata.get("system", "general")
        content_type = doc.metadata.get("content_type", "text")
        context_parts.append(
            f"--- Chunk {i} [Source: {source}, Page: {page}, "
            f"System: {system}, Type: {content_type}, "
            f"Relevance: {score:.2f}] ---\n{doc.page_content}"
        )
    return "\n\n".join(context_parts)


def get_source_references(results: list) -> list:
    sources = []
    seen = set()
    for doc, score in results:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "system": doc.metadata.get("system", "general"),
                "content_type": doc.metadata.get("content_type", "text"),
                "relevance": round(score, 2)
            })
    return sources


if __name__ == "__main__":
    test_query = "hydraulic pump not building pressure"
    print(f"Test Query: {test_query}\n")
    results = retrieve(test_query)
    if results:
        for i, (doc, score) in enumerate(results, 1):
            print(f"Result {i} (score: {score:.2f}):")
            print(f"  Source: {doc.metadata['source']}, Page: {doc.metadata['page']}")
            print(f"  Content: {doc.page_content[:200]}...\n")
    else:
        print("No results. Run ingest.py first!")
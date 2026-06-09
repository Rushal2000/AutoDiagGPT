import streamlit as st
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chain import ask_with_fallback
from ingest import run_ingestion
from config import LLM_PROVIDER

st.set_page_config(page_title="AutoDiagGPT", page_icon="wrench", layout="wide")

with st.sidebar:
    st.title("AutoDiagGPT")
    st.caption("RAG Troubleshooting Assistant for Off-Highway Vehicles")
    st.divider()

    system_filter = st.selectbox(
        "Filter by Vehicle System",
        options=["all", "hydraulics", "bms", "motor", "motor_controller",
                 "can", "sensors", "steering", "brakes", "safety", "general"],
        index=0
    )
    st.divider()

    st.markdown("### LLM Provider")
    provider_info = {
        "groq": "Groq - Llama 3.3 70B\n14,400 req/day FREE | ~315 TPS",
        "gemini": "Gemini - Gemini 2.5 Flash\n1,500 req/day FREE",
    }
    st.markdown(provider_info.get(LLM_PROVIDER, "Unknown"))
    st.caption("Auto-fallback: Groq -> Gemini")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Re-ingest", use_container_width=True):
            with st.spinner("Ingesting..."):
                run_ingestion()
            st.success("Done!")
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = ""
            st.rerun()

    st.divider()
    st.markdown("### Architecture")
    st.markdown("- Embeddings: nomic-embed-text (local)\n"
                "- Vector DB: ChromaDB (local)\n"
                "- LLM: Free API (cloud)\n"
                "- RAM: ~2.5 GB on Jetson")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

st.title("AutoDiagGPT")
st.markdown("*Describe the fault, enter an error code, or ask about any troubleshooting procedure.*")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Source References"):
                for src in message["sources"]:
                    st.markdown(
                        f"- **{src['source']}** - Page {src['page']} "
                        f"(System: {src['system']}, Type: {src['content_type']}, "
                        f"Relevance: {src['relevance']})"
                    )
        if "provider" in message:
            st.caption(f"Provider: {message['provider']}")

if prompt := st.chat_input("Describe the fault or enter an error code..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & generating response..."):
            start_time = time.time()
            try:
                result = ask_with_fallback(
                    query=prompt,
                    system_filter=system_filter if system_filter != "all" else None,
                    chat_history=st.session_state.chat_history
                )
                elapsed = time.time() - start_time
                answer = result["answer"]
                sources = result["sources"]
                provider = result["provider"]

                st.markdown(answer)
                if sources:
                    with st.expander("Source References"):
                        for src in sources:
                            st.markdown(
                                f"- **{src['source']}** - Page {src['page']} "
                                f"(System: {src['system']}, Type: {src['content_type']})")
                st.caption(
                    f"Provider: {provider} | "
                    f"Chunks: {result['num_chunks_retrieved']} | "
                    f"Time: {elapsed:.1f}s")

                st.session_state.chat_history += f"\nTechnician: {prompt}\nAutoDiagGPT: {answer}\n"
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "provider": provider})
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("1. Make sure Ollama is running: ollama serve\n"
                        "2. Check your API key in .env\n"
                        "3. Verify internet connection")

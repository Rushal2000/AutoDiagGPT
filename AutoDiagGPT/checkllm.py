"""
check_llm.py — Quick sanity check for Groq API
Run: python check_llm.py
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found in .env file.")
    exit(1)

print("=" * 50)
print("  AutoDiagGPT — Groq LLM Check")
print("=" * 50)
print(f"Model : {GROQ_MODEL}")
print(f"Key   : {GROQ_API_KEY[:8]}{'*' * 20}")
print("-" * 50)

try:
    llm = ChatGroq(
        model_name=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=64,
    )
    response = llm.invoke("Reply with exactly: AutoDiagGPT LLM is working.")
    print(f"Response: {response.content}")
    print("-" * 50)
    print("SUCCESS: Groq API is connected and working!")

except Exception as e:
    print(f"FAILED : {e}")
    print("-" * 50)
    print("Check: API key is correct | Internet is available")
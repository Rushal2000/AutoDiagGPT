import os
import re
import json
import fitz  # PyMuPDF
fitz.TOOLS.set_icc(False)
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import (
    DATA_DIR, VECTORSTORE_DIR, OCR_FILE, CAPTION_FILE,
    EMBEDDING_MODEL, OLLAMA_BASE_URL, CHUNK_SIZE, CHUNK_OVERLAP,
    detect_system
)


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    return text.strip()


def load_pdf_text() -> list:
    docs = []
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"Warning: No PDF files found in {DATA_DIR}")
        return docs

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        print(f"Processing: {pdf_file}")
        page_count = 0
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, 1):
                text = page.get_text("text")
                if text and text.strip():
                    docs.append({
                        "text": clean_text(text), "source": pdf_file,
                        "page": page_num, "content_type": "text"
                    })
                    page_count += 1
            doc.close()

            if page_count == 0:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text()
                        if text and text.strip():
                            docs.append({
                                "text": clean_text(text), "source": pdf_file,
                                "page": page_num, "content_type": "text"
                            })
                            page_count += 1

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            table_text = ""
                            for row in table:
                                row_clean = [str(cell) if cell else "" for cell in row]
                                table_text += " | ".join(row_clean) + "\n"
                            if table_text.strip():
                                docs.append({
                                    "text": f"[TABLE from {pdf_file}, Page {page_num}]\n{table_text}",
                                    "source": pdf_file, "page": page_num, "content_type": "table"
                                })
            print(f"   Extracted {page_count} pages")
        except Exception as e:
            print(f"   Error: {e}")

    print(f"\nTotal text pages: {len(docs)}")
    return docs


def load_ocr_results() -> list:
    if not os.path.exists(OCR_FILE):
        print("No OCR results found, skipping...")
        return []
    with open(OCR_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} OCR results")
    return data


def load_diagram_captions() -> list:
    if not os.path.exists(CAPTION_FILE):
        print("No diagram captions found, skipping...")
        return []
    with open(CAPTION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} diagram captions")
    return data


def build_all_documents() -> list:
    all_docs = []

    pdf_pages = load_pdf_text()
    for page in pdf_pages:
        all_docs.append(Document(
            page_content=page["text"],
            metadata={"source": page["source"], "page": page["page"],
                      "content_type": page["content_type"],
                      "system": detect_system(page["text"])}
        ))

    ocr_results = load_ocr_results()
    for ocr in ocr_results:
        enriched = f"[DIAGRAM TEXT — from {ocr['source_pdf']}, Page {ocr['page']}]\n{ocr['ocr_text']}"
        all_docs.append(Document(
            page_content=enriched,
            metadata={"source": ocr.get("source_pdf", "unknown"), "page": ocr.get("page", "?"),
                      "content_type": "diagram_ocr", "source_image": ocr.get("source_image", ""),
                      "system": detect_system(ocr["ocr_text"])}
        ))

    captions = load_diagram_captions()
    for cap in captions:
        enriched = f"[DIAGRAM DESCRIPTION — from {cap['source_pdf']}, Page {cap['page']}]\n{cap['caption']}"
        if "components" in cap:
            enriched += f"\nKey components: {', '.join(cap['components'])}"
        all_docs.append(Document(
            page_content=enriched,
            metadata={"source": cap.get("source_pdf", "unknown"), "page": cap.get("page", "?"),
                      "content_type": "diagram_caption", "source_image": cap.get("source_image", ""),
                      "system": detect_system(cap["caption"])}
        ))

    print(f"\nTotal documents combined: {len(all_docs)}")
    return all_docs


def chunk_documents(documents: list) -> list:

    # ===== BEFORE CHUNKING STATS =====
    original_chars = sum(len(doc.page_content) for doc in documents)
    original_words = sum(len(doc.page_content.split()) for doc in documents)

    print("\n========== BEFORE CHUNKING ==========")
    print(f"Documents      : {len(documents)}")
    print(f"Characters     : {original_chars:,}")
    print(f"Words          : {original_words:,}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for doc_item in documents:
        chunks = splitter.split_documents([doc_item])
        all_chunks.extend(chunks)

    # ===== AFTER CHUNKING STATS =====
    chunk_chars = sum(len(chunk.page_content) for chunk in all_chunks)
    chunk_words = sum(len(chunk.page_content.split()) for chunk in all_chunks)

    print("\n========== AFTER CHUNKING ==========")
    print(f"Chunks         : {len(all_chunks)}")
    print(f"Characters     : {chunk_chars:,}")
    print(f"Words          : {chunk_words:,}")

    print("\n========== CHUNKING IMPACT ==========")
    print(f"Character Growth : {chunk_chars - original_chars:,}")
    print(f"Word Growth      : {chunk_words - original_words:,}")

    if original_chars > 0:
        growth = ((chunk_chars - original_chars) / original_chars) * 100
        print(f"Growth %         : {growth:.2f}%")

    # Sample chunk size statistics
    chunk_lengths = [len(chunk.page_content) for chunk in all_chunks]

    print("\n========== CHUNK STATS ==========")
    print(f"Min Chunk Size : {min(chunk_lengths)} chars")
    print(f"Max Chunk Size : {max(chunk_lengths)} chars")
    print(f"Avg Chunk Size : {sum(chunk_lengths)/len(chunk_lengths):.2f} chars")

    return all_chunks


def create_vectorstore(chunks: list) -> Chroma:
    print(f"\nGenerating embeddings using '{EMBEDDING_MODEL}' via Ollama...")
    embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
    num_gpu=0   # force CPU, no VRAM used — passed directly, not via model_kwargs
)
    
    # Remove old vector store
    if os.path.exists(VECTORSTORE_DIR):
        import shutil
        shutil.rmtree(VECTORSTORE_DIR)
        print("Cleared old vector store")

    BATCH_SIZE = 10  # Safe for Jetson

    vectorstore = None

    total_chunks = len(chunks)
    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Total chunks: {total_chunks}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Total batches: {total_batches}")

    for batch_num in range(total_batches):

        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_chunks)

        batch = chunks[start_idx:end_idx]

        print(
            f"\nProcessing Batch {batch_num + 1}/{total_batches}"
            f" | Chunks {start_idx + 1}-{end_idx}"
        )

        try:
            if vectorstore is None:

                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=VECTORSTORE_DIR
                )

            else:

                vectorstore.add_documents(batch)

            print(
                f"✓ Batch {batch_num + 1} completed "
                f"({len(batch)} chunks)"
            )

        except Exception as e:

            print(
                f"✗ Batch {batch_num + 1} failed:\n{e}"
            )

            # Continue with remaining batches
            continue

    if vectorstore is not None:
        # vectorstore.persist()  # no longer needed/available in newer langchain-chroma
        pass

    print("\n======================================")
    print("Vector Store Creation Complete")
    print("======================================")
    print(f"Stored {total_chunks} chunks")
    print(f"Saved to: {VECTORSTORE_DIR}")

    return vectorstore


def run_ingestion():
    print("=" * 60)
    print("  AutoDiagGPT — Document Ingestion (Text + Diagrams)")
    print("=" * 60)
    documents = build_all_documents()
    if not documents:
        print("No documents to process!")
        return
    chunks = chunk_documents(documents)
    create_vectorstore(chunks)
    print("\nIngestion complete! Vector store is ready.")


if __name__ == "__main__":
    run_ingestion()
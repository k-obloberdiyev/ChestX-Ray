"""
Automated Clinical PDF Protocol Ingestion Engine for AvicennaX RAG.
Extracts text from official PDF medical standards and updates the local vector store.
"""
import os
import re
import json
import uuid
import logging
from typing import Dict, Any, List, Optional
import pypdfium2 as pdfium

from backend.core.rag.vector_store import get_vector_store, KNOWLEDGE_DIR

logger = logging.getLogger("rag_pdf_ingest")


def extract_text_from_pdf(pdf_path_or_bytes: Any) -> str:
    """
    Extract clean textual content from PDF file path or raw bytes using pypdfium2.
    """
    try:
        if isinstance(pdf_path_or_bytes, (bytes, bytearray)):
            pdf = pdfium.PdfDocument(pdf_path_or_bytes)
        else:
            pdf = pdfium.PdfDocument(str(pdf_path_or_bytes))

        pages_text = []
        for i, page in enumerate(pdf):
            textpage = page.get_textpage()
            page_text = textpage.get_text_range().strip()
            if page_text:
                pages_text.append(page_text)

        full_text = "\n\n".join(pages_text)
        return full_text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}", exc_info=True)
        raise RuntimeError(f"PDF extraction error: {e}") from e


def structure_protocol_content(
    raw_text: str,
    disease: str,
    protocol_title: Optional[str] = None,
    protocol_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Structure raw text into a standardized AvicennaX RAG protocol document.
    """
    pid = protocol_id or f"PROTOCOL-PDF-{uuid.uuid4().hex[:6].upper()}"
    title = protocol_title or f"SSV Milliy Klinik Standarti: {disease} (2025)"

    # Clean text
    clean_text = re.sub(r'[ \t]+', ' ', raw_text).strip()

    # Build tags
    disease_tokens = [d.lower() for d in re.findall(r'\w+', disease) if len(d) > 2]
    tags = list(set(["ssv", "standart", "protokol", "dori", "doza", disease.lower()] + disease_tokens))

    # Format content with clear medical structure
    formatted_content = f"### 📋 Rasmiy Klinik Standart & Ko'rsatmalar ({title})\n{clean_text}"

    doc = {
        "id": pid,
        "disease": disease,
        "title_uz": title,
        "title_ru": f"Клинический Стандарт: {disease} (Минздрав РУз)",
        "title_en": f"Official Clinical Standard for {disease}",
        "content_uz": formatted_content,
        "content_ru": formatted_content,
        "content_en": formatted_content,
        "tags": tags,
        "source": "PDF Ingested Standard",
        "raw_text_length": len(raw_text)
    }
    return doc


def ingest_pdf_to_knowledge(
    pdf_path_or_bytes: Any,
    disease: str,
    protocol_title: Optional[str] = None,
    protocol_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete pipeline to ingest a PDF clinical protocol into the local RAG knowledge base.
    
    1. Extract text via pypdfium2.
    2. Format structured document.
    3. Save JSON in rag/knowledge/.
    4. Automatically reload and re-index vector store in memory.
    """
    raw_text = extract_text_from_pdf(pdf_path_or_bytes)
    if not raw_text or len(raw_text.strip()) < 10:
        raise ValueError("The provided PDF does not contain extractable text or is empty.")

    doc = structure_protocol_content(
        raw_text=raw_text,
        disease=disease,
        protocol_title=protocol_title,
        protocol_id=protocol_id
    )

    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    safe_filename = f"ingested_{re.sub(r'[^a-zA-Z0-9_]', '_', disease.lower())}_{doc['id'].lower()}.json"
    file_path = os.path.join(KNOWLEDGE_DIR, safe_filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([doc], f, ensure_ascii=False, indent=2)

    logger.info(f"Saved ingested protocol to {file_path}")

    # Immediately reload and re-index the vector store
    store = get_vector_store()
    store.load_and_index()
    logger.info("Vector store re-indexed with new PDF protocol.")

    return {
        "status": "success",
        "protocol_id": doc["id"],
        "disease": disease,
        "title": doc["title_uz"],
        "file_saved": file_path,
        "text_length": len(raw_text)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest PDF clinical standard into AvicennaX RAG")
    parser.add_argument("--pdf", required=True, help="Path to PDF protocol file")
    parser.add_argument("--disease", required=True, help="Target disease (e.g. Pneumonia, Asthma, COPD)")
    parser.add_argument("--title", required=False, help="Custom protocol title")

    args = parser.parse_args()
    res = ingest_pdf_to_knowledge(args.pdf, args.disease, args.title)
    print("Ingestion Result:", json.dumps(res, indent=2, ensure_ascii=False))

"""
Unit Tests for Local RAG Engine & Vector Store.
"""
from backend.core.rag.vector_store import LocalVectorStore, get_vector_store
from backend.core.rag.rag_engine import query_rag_assistant, build_rag_report


def test_vector_store_indexing_and_search():
    store = get_vector_store()
    results = store.search("Pnevmoniya", top_k=1, lang="uz")
    assert len(results) > 0
    doc, score = results[0]
    assert doc["disease"] == "Pneumonia"
    assert score > 0.0


def test_rag_query_assistant_grounding():
    result = query_rag_assistant(
        patient_id="PAT-TEST",
        diagnosis="Pneumonia",
        user_message="pnevmoniya dori va isitma",
        lang="uz"
    )
    assert result["is_rag_grounded"] is True
    assert len(result["citations"]) > 0
    assert "Pneumonia" in result["citations"][0]["disease"]


def test_rag_multilingual_support():
    result_ru = query_rag_assistant(
        patient_id="PAT-TEST",
        diagnosis="Pneumothorax",
        user_message="пневмоторакс экстренный",
        lang="ru"
    )
    assert result_ru["is_rag_grounded"] is True
    assert "Пневмоторакс" in result_ru["message"] or "Клинический" in result_ru["message"]


def test_qwen_integration_fallback():
    from backend.core.rag.qwen_llm import generate_qwen_response
    res = generate_qwen_response(
        user_query="Pnevmoniya bo'yicha yo'riqnoma",
        context_chunks=["Pnevmoniya klinik davolash protokoli"],
        diagnosis="Pneumonia",
        lang="uz"
    )
    assert "status" in res
    assert "source" in res


def test_copd_protocol_retrieval():
    """Test retrieval of official SSV COPD Protocol & Standard (Order 180, 2025)."""
    store = get_vector_store()
    results = store.search("SOO'K GOLD A-B-E Salbutamol Tiotropiy", top_k=2, lang="uz")
    assert len(results) > 0
    assert any("SOO'K" in r[0].get("disease", "") or "COPD" in r[0].get("disease", "") for r in results)

    # Test query assistant returns SSV 180 grounded content
    rag_res = query_rag_assistant(
        patient_id="MX-TEST",
        diagnosis="Surunkali Obstruktiv O'pka Kasalligi (SOO'K / XOBL / COPD)",
        user_message="SOO'Kda qanday dori vositalari va dozalar buyuriladi?",
        lang="uz"
    )
    assert rag_res["is_rag_grounded"] is True
    assert "180" in rag_res["message"] or "SSV" in rag_res["message"] or "Salbutamol" in rag_res["message"]


def test_pdf_protocol_ingestion_and_search(tmp_path):
    """Test offline extraction of clinical PDF protocol and immediate indexing in vector store."""
    import io
    from backend.core.rag.ingest_pdf import ingest_pdf_to_knowledge
    from backend.core.rag.vector_store import get_vector_store
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.drawString(100, 750, "SSV Klinik Standarti: Bronxial Astma (Buyruq 180)")
    c.drawString(100, 730, "Salbutamol 100 mkg ingalyatsiya va Budesonid 200 mkg.")
    c.drawString(100, 710, "SpO2 < 90 bo'lsa shoshilinch statsionar yordam.")
    c.save()
    pdf_bytes = pdf_buffer.getvalue()

    file_to_clean = None
    try:
        res = ingest_pdf_to_knowledge(
            pdf_path_or_bytes=pdf_bytes,
            disease="Asthma",
            protocol_title="SSV Standarti: Astma (2025)",
            protocol_id="PROTOCOL-TEST-ASTHMA"
        )
        file_to_clean = res.get("file_saved")
        assert res["status"] == "success"
        assert res["protocol_id"] == "PROTOCOL-TEST-ASTHMA"

        # Verify vector store immediately retrieves it
        store = get_vector_store()
        results = store.search("Salbutamol Budesonid Astma", top_k=1, lang="uz")
        assert len(results) > 0
        assert results[0][0]["disease"] == "Asthma"
    finally:
        import os
        if file_to_clean and os.path.exists(file_to_clean):
            os.remove(file_to_clean)
        store = get_vector_store()
        store.load_and_index()



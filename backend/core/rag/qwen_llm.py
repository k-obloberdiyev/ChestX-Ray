import os
import json
import logging
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger("qwen_llm")

# Configuration via Environment Variables
QWEN_SERVER_URL = os.getenv("QWEN_SERVER_URL", "http://localhost:11434")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen2.5")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_TIMEOUT = int(os.getenv("QWEN_TIMEOUT", "10"))


def generate_qwen_response(
    user_query: str,
    context_chunks: List[str],
    diagnosis: str = "Norma",
    lang: str = "uz"
) -> Dict[str, Any]:
    """
    Sends RAG retrieved context + query to Qwen LLM.
    If Qwen server is available, returns Qwen-generated medical response.
    Otherwise, gracefully falls back to structured RAG synthesis.
    """
    system_instructions = {
        "uz": (
            "Siz AvicennaX AI - O'zbekiston Respublikasi Sog'liqni saqlash vazirligi (SSV) Milliy Klinik Protokollari "
            "va Standartlari asosida ishlovchi klinik maslahatchi AI yordamchisiz.\n\n"
            "QAT'IY QOIDALAR VA STANDARTGA BOG'LIQLIK:\n"
            "1. FAQAT taqdim etilgan rasmiy klinik protokol kontekstiga asoslanib javob bering. Kontekstda bo'lmagan dori yoki davolash usullarini aslo tavsiya qilmang (Zero Hallucination).\n"
            "2. MANBA: Javobni doimo birinchi qatordan rasmiy manba bilan boshlang: '📌 **Manba**: O'zbekiston SSV Milliy Klinik Protokoli (180-sonli buyruq / Standart)'\n"
            "3. SARLAVHALAR (HEADERS): Javobingizni quyidagi aniq sarlavhalar ('###') bo'yicha ajrating:\n"
            "   ### 📋 1. Rasmiy Standart Mezonlari\n"
            "   ### 💊 2. Tasdiqlangan Dori Vositalari va Dozalash Sxemalari\n"
            "   ### 🚨 3. Shoshilinch Ko'rsatmalar va Vrach Nazorati\n"
            "4. DORI-DORILAR: Har bir dori nomini **Qalin** (bold) yozing, dozasini va qabul qilish tartibini faqat nuqtalar ('•') bilan ko'rsating:\n"
            "   • **Dori nomi** (Guruh): doza, qabul qilish tartibi, kursi.\n"
            "5. Agar savolga javob berilgan protokolda mavjud bo'lmasa: 'Ushbu ma'lumot SSV klinik standartida ko'rsatilmagan. Mutaxassis vrach-pulmonolog ko'rigiga murojaat qiling.' deb javob bering."
        ),
        "ru": (
            "Вы — медицинский ИИ-ассистент AvicennaX, строго следующий Национальным клиническим протоколам Минздрава РУз "
            "(Приказ №180 / Национальные Стандарты).\n\n"
            "СТРОГИЕ ПРАВИЛА И ЗАВИСИМОСТЬ ОТ ПРОТОКОЛА:\n"
            "1. Отвечайте ТОЛЬКО на основе предоставленного контекста протокола. Запрещено добавлять не указанные в протоколе препараты.\n"
            "2. ИСТОЧНИК: Всегда начинайте с первой строки: '📌 **Источник**: Национальный клинический протокол Минздрава РУз (Приказ №180)'\n"
            "3. СТРУКТУРА (HEADERS):\n"
            "   ### 📋 1. Официальные Диагностические Критерии\n"
            "   ### 💊 2. Утвержденная Фармакотерапия и Схемы Дозирования\n"
            "   ### 🚨 3. Экстренные Указания и Врачебный Контроль\n"
            "4. ПРЕПАРАТЫ: Выделяйте препараты **Жирным** шрифтом с точной дозировкой маркерами ('•'):\n"
            "   • **Препарат** (Группа): дозировка, режим приема, курс.\n"
            "5. Если информация отсутствует в протоколе, явно укажите: 'Данная информация не содержится в протоколе Минздрава. Необходима консультация профильного специалиста.'"
        ),
        "en": (
            "You are the AvicennaX AI Clinical Assistant, operating strictly according to the Uzbekistan MOH National Clinical Protocols "
            "and Standards (Order No. 180).\n\n"
            "STRICT PROTOCOL GROUNDING RULES:\n"
            "1. Answer ONLY based on the provided official clinical standard context. Do NOT invent unlisted drugs or dosages.\n"
            "2. SOURCE: Always begin on line 1 with: '📌 **Source**: Uzbekistan MOH National Clinical Protocol (Order No. 180)'\n"
            "3. SECTION HEADERS:\n"
            "   ### 📋 1. Official Diagnostic Criteria\n"
            "   ### 💊 2. Approved Pharmacotherapy & Dosage Regimens\n"
            "   ### 🚨 3. Emergency Directives & Clinical Oversight\n"
            "4. MEDICATIONS: Bold drug names (**Drug Name**), specify exact dose, route, and duration strictly using bullets ('•'):\n"
            "   • **Drug Name** (Class): dosage, frequency, duration.\n"
            "5. If not found in the context, explicitly state: 'This information is not covered in the indexed clinical protocol. Please consult an attending physician.'"
        )
    }

    sys_prompt = system_instructions.get(lang, system_instructions["uz"])

    formatted_context = "\n\n".join(context_chunks) if context_chunks else "Klinik protokol va Milliy Standart konteksti topilmadi."

    user_prompt = (
        f"Klinik Tashxis: {diagnosis}\n"
        f"Rasmiy SSV Milliy Klinik Standarti Konteksti:\n{formatted_context}\n\n"
        f"Shifokor Savoli: {user_query}\n\n"
        f"Iltimos, javobni QISQA va FAQAT yuqoridagi SSV standarti asosida bering."
    )

    # 1. Try Ollama Native API (http://localhost:11434/api/chat)
    ollama_url = f"{QWEN_SERVER_URL.rstrip('/')}/api/chat"
    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }

    headers = {"Content-Type": "application/json"}
    if QWEN_API_KEY:
        headers["Authorization"] = f"Bearer {QWEN_API_KEY}"

    try:
        response = requests.post(ollama_url, json=payload, headers=headers, timeout=QWEN_TIMEOUT)
        if response.status_code == 200:
            res_json = response.json()
            reply_text = res_json.get("message", {}).get("content", "").strip()
            if reply_text:
                return {
                    "source": f"Qwen LLM ({QWEN_MODEL_NAME}) via Ollama",
                    "status": "success",
                    "text": reply_text
                }
    except Exception as e:
        logger.debug(f"Ollama Qwen endpoint not reachable at {ollama_url}: {e}")

    # 2. Try OpenAI-compatible endpoint (http://localhost:11434/v1/chat/completions)
    v1_url = f"{QWEN_SERVER_URL.rstrip('/')}/v1/chat/completions"
    payload_v1 = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }

    try:
        response_v1 = requests.post(v1_url, json=payload_v1, headers=headers, timeout=QWEN_TIMEOUT)
        if response_v1.status_code == 200:
            res_v1 = response_v1.json()
            reply_text = res_v1["choices"][0]["message"]["content"].strip()
            if reply_text:
                return {
                    "source": f"Qwen LLM ({QWEN_MODEL_NAME}) via v1/completions",
                    "status": "success",
                    "text": reply_text
                }
    except Exception as e:
        logger.debug(f"V1 Qwen endpoint not reachable at {v1_url}: {e}")

    # 3. Fallback when Qwen server is offline
    return {
        "source": "Local RAG Vector Engine (Qwen server offline)",
        "status": "fallback",
        "text": None
    }

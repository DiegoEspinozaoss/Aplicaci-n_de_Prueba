import os
import time
from contextlib import contextmanager
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# ================================
#         PARÁMETROS
# ================================

PARAMS = {
    "CHUNK_SIZE": 4000,
    "CHUNK_OVERLAP": 500,
    "TOP_K": 8,

    "EMBED_MODEL": "text-embedding-3-large",
    "EMBED_API_KEY": os.getenv("EMBED_API_KEY"),

    "LLM_MODEL": "gpt-4o-mini",
    "LLM_API_KEY": os.getenv("LLM_API_KEY"),

    "LLM_TEMP": 0,
    "LLM_MAX_TOKENS": 6000   # seguro para Azure
}

# ================================
#        TEMPLATE RAG
# ================================

RAG_TEMPLATE = """
Eres un asistente que responde SOLO en español y SOLO con base en el CONTEXTO entregado.
Devuelve solo la información pedida, sin texto adicional.

Si la respuesta no está explícitamente en el contexto, di: "No está disponible en el PDF".

CONTEXTO:
{context}

PREGUNTA:
{question}

Respuesta breve y precisa:
"""

# ================================
#     PREGUNTAS DEFAULT
# ================================

PREGUNTAS = ["¿Quién es el emisor?"]
CLAVES = {"emisor": "emisor"}



# ================================
#         UTILIDADES
# ================================

@contextmanager
def cronometro(etiqueta: str):
    t0 = time.perf_counter()
    print(f"[{etiqueta}] iniciando...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[{etiqueta}] listo en {dt:.2f} s")


def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source','?')} | p.{d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )

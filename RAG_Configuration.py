import time
from contextlib import contextmanager
from typing import List
import pymupdf4llm
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

PARAMS = {
    "CHUNK_SIZE": 4000,#tamaño de chunk en caracteres
    "CHUNK_OVERLAP": 500,# overlap entre chunks en caracteres
    "TOP_K": 8,# número de documentos a recuperar del vectorstore      
    "EMBED_MODEL": "text-embedding-3-large",#sentence-transformers/all-mpnet-base-v2", # alternativa: "sentence-transformers/all-MiniLM-L6-v2"
    "EMBED_BASE_URL": "https://api.openai.com/v1",
    "EMBED_API_KEY": os.getenv("EMBED_API_KEY"),
    "LLM_MODEL": "gpt-4o-mini","LLM_MODEL": "gpt-4o-mini",#"mistral-7b-instruct", # alternativa: "",meta-llama-3.1-8b-instruct
    "LLM_BASE_URL": "https://api.openai.com/v1",#"http://127.0.0.1:1234/v1", URL base del LLM local
    "LLM_API_KEY": os.getenv("LLM_API_KEY"),
    "LLM_TEMP": 0,
    "LLM_MAX_TOKENS": 10000,#maximo de tokens en la respuesta del LLM
}


RAG_TEMPLATE = """
Eres un asistente que responde SOLO en español y SOLO con base en el CONTEXTO entregado.
Devuelve solo la información pedida, sin texto adicional.

Si la respuesta no está explícitamente en el contexto, di: "No está disponible en el PDF".

Cuando te pida el **emisor del PDF**, busca principalmente en el **encabezado o pie de página**. En general, es un banco.
Siempre es diferente del cliente o asesor financiero que recibe la cartola. No es una persona
Dentro de su nombre puede contener "ban" como "banchile", "scotiabank", "security", "bice", "jp morgan", etc.

Cuando te pida la **fecha de cierre del PDF**, busca frases como “Fecha de corte”, “Cierre al”, “Fecha de emisión”, etc. 
Quiero que me lo des en formato DD/MM/AAAA. Si no está en ese formato, conviértelo. 
Recuerda que el DD es el día (01 a 31), MM es el mes (01 a 12) y AAAA es el año (4 dígitos).

Cuando te pida el **nombre de la empresa a quien se le envía el documento**, en general está en el encabezado del PDF y puede
contener "inversiones" o "asesorías" en su nombre. Es diferente del emisor del PDF. Por lo general, la cartola se puede dirigir a 
dicha empresa con pronombres personales como "Estimado cliente", "A quien corresponda", etc. Recuerda no confundir el nombre del cliente con el nombre del asesor financiero o banco emisor.

Asume que puede haber una o más cuentas asociadas a dicha empresa. Devuélvelos todos, separados por comas (no los confundas con dinero).
Cuando te pida el número de cuenta, busca frases como “Cuenta:”, “N° de cuenta:”, “Número de cuenta:”, etc. 

CONTEXTO:
{context}

PREGUNTA:
{question}

Respuesta útil y concisa:
"""

PREGUNTAS = [
        # "¿Quién es el emisor de esta cartola?",
        # "¿Cuál es la fecha de cierre de la cartola?",
        # "¿Cuál es el nombre de la empresa cliente?",
        # "Dame todos los números de cuenta asociados a la empresa cliente.",
    ]

CLAVES = {
    PREGUNTAS[0]: "emisor",
    # PREGUNTAS[1]: "fecha_cierre",
    # PREGUNTAS[2]: "cliente",
    # PREGUNTAS[3]: "numero_cuenta",
    # PREGUNTAS[4]: "patrimonio",
    # PREGUNTAS[0]: "movimientos",
}

@contextmanager
def cronometro(etiqueta: str):
    """Cronómetro simple para medir duración de bloques de código."""
    t0 = time.perf_counter()
    print(f"[{etiqueta}] iniciando...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[{etiqueta}] listo en {dt:0.2f} s")

def format_docs(docs: List[Document]) -> str:
    """
    Formatea una lista de Document en texto plano con metadatos de origen y página.
    """
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source','?')} | p.{d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )

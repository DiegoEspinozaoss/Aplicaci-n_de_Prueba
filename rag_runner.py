# rag_runner.py

import os
import pathlib
import time
import json
import numpy as np
from typing import Dict

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from RAG_Configuration import (
    PARAMS,
    RAG_TEMPLATE,
    PREGUNTAS,
    CLAVES,
    format_docs,
    cronometro
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tiktoken


tok_encoder = tiktoken.encoding_for_model(PARAMS["EMBED_MODEL"])
tok_llm = tiktoken.encoding_for_model(PARAMS["LLM_MODEL"])

def tokens_llm(text):
    return len(tok_llm.encode(text))


def obtener_markdown_por_pagina(pdf_file: str):
    import fitz
    doc = fitz.open(pdf_file)
    markdown_pages = []
    for i in range(len(doc)):
        md = pymupdf4llm.to_markdown(
            doc=pdf_file,
            pages=[i],
            table_strategy="lines",
            fontsize_limit=3,
            margins=0.1,
            ignore_images=True,
            ignore_graphics=True,
            page_separators=False,
        )
        if md.strip():
            markdown_pages.append(Document(
                page_content=md,
                metadata={"source": pdf_file, "page": i + 1}
            ))
    return markdown_pages


# ============================================================
#                 FUNCIÓN PRINCIPAL RAG
# ============================================================

def run_rag(pdf_path: str) -> Dict:
    """Ejecuta TODO el pipeline RAG pero SOLO para un PDF y retorna resultados."""

    base_name = pathlib.Path(pdf_path).stem
    print(f"\n=== Procesando {pdf_path} ===")

    # Parámetros
    CHUNK_SIZE = PARAMS["CHUNK_SIZE"]
    CHUNK_OVERLAP = PARAMS["CHUNK_OVERLAP"]
    TOP_K = PARAMS["TOP_K"]

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Modelo LLM
    llm = ChatOpenAI(
        model=PARAMS["LLM_MODEL"],
        base_url=PARAMS["LLM_BASE_URL"],
        api_key=PARAMS["LLM_API_KEY"],
        temperature=PARAMS["LLM_TEMP"],
        max_tokens=PARAMS["LLM_MAX_TOKENS"]
    )

    parser = StrOutputParser()
    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # ============================================================
    #            1. PDF → markdown por página
    # ============================================================
    docs_por_pagina = obtener_markdown_por_pagina(pdf_path)

    # ============================================================
    #       2. Split a chunks para embeddings
    # ============================================================
    chunks = []
    for d in docs_por_pagina:
        for part in splitter.split_text(d.page_content):
            chunks.append(Document(page_content=part, metadata=d.metadata))

    # ============================================================
    #              3. Embeddings + FAISS
    # ============================================================
    with cronometro(f"Creación del índice FAISS para {base_name}"):
        emb = OpenAIEmbeddings(
            model=PARAMS["EMBED_MODEL"],
            api_key=PARAMS["EMBED_API_KEY"],
            base_url=PARAMS["EMBED_BASE_URL"],
            dimensions=3072
        )
        vs = FAISS.from_documents(chunks, emb)

    # ============================================================
    #          4. Preguntas del RAG (tu lista PREGUNTAS)
    # ============================================================
    respuestas = {}
    recall_scores = {}

    for q in PREGUNTAS:
        print(f"\n--- Pregunta: {q}")

        # Vector search
        results_with_scores = vs.similarity_search_with_score(q, k=TOP_K)

        # Ordenar por score descendente (mayor relevancia)
        results_with_scores.sort(key=lambda x: x[1], reverse=True)

        docs = [doc for doc, score in results_with_scores]

        if not docs:
            respuestas[q] = "No está disponible en el PDF."
            recall_scores[q] = 0.0
            continue

        contexto = format_docs(docs)

        prompt_text = (
            f"{prompt.format(context=contexto, question=q)}\n\n"
            "Devuelve SOLO la respuesta sin texto adicional."
        )

        # Llamada al LLM
        llm_resp = llm.invoke(prompt_text)
        ans = parser.invoke(llm_resp).strip().strip('"')

        respuestas[q] = ans

        # Verificación muy básica del recall
        encontrado = any(ans.lower() in d.page_content.lower() for d in docs)
        recall_scores[q] = 1.0 if encontrado else 0.0

    # ============================================================
    #                        RETORNO
    # ============================================================
    salida = {
        "archivo": os.path.basename(pdf_path),
        "respuestas": respuestas,
        "recall_por_pregunta": recall_scores,
        "recall_promedio": round(sum(recall_scores.values()) / len(recall_scores), 3)
    }

    # PARA AZURE: devolver JSON simple al front-end
    return salida


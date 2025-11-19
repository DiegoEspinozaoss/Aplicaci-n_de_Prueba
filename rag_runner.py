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


def run_rag(pdf_path: str) -> Dict:
    """Ejecuta TODO el pipeline RAG pero SOLO para un PDF."""
    
    base_name = pathlib.Path(pdf_path).stem
    print(f"\n=== Procesando {pdf_path} ===")
    
    CHUNK_SIZE = PARAMS["CHUNK_SIZE"]
    CHUNK_OVERLAP = PARAMS["CHUNK_OVERLAP"]
    TOP_K = PARAMS["TOP_K"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    llm = ChatOpenAI(
        model=PARAMS["LLM_MODEL"],
        base_url=PARAMS["LLM_BASE_URL"],
        api_key=PARAMS["LLM_API_KEY"],
        temperature=PARAMS["LLM_TEMP"],
        max_tokens=PARAMS["LLM_MAX_TOKENS"],
    )

    parser = StrOutputParser()
    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # --- 1: convertir PDF a markdown ---
    docs_por_pagina = obtener_markdown_por_pagina(pdf_path)

    # --- 2: crear chunks ---
    chunks = []
    for d in docs_por_pagina:
        for part in splitter.split_text(d.page_content):
            chunks.append(Document(page_content=part, metadata=d.metadata))

    # --- 3: FAISS + embeddings --

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import os
from transformers import logging
logging.set_verbosity_info()

# ========================
# CONFIG
# ========================
# Modelos utilizados:
# - EMBED_MODEL: para convertir preguntas y documentos en vectores numéricos (embeddings).
# - QA_MODEL: modelo extractivo para responder preguntas en español a partir de un contexto.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
QA_MODEL = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"  # modelo extractivo en español

# ========================
# CARGA DE EMBEDDINGS
# ========================
print("🔍 Cargando modelo de embeddings...")
embedding_model = SentenceTransformer(
    EMBED_MODEL,
    device='cuda' if torch.cuda.is_available() else 'cpu' # Usa GPU si está disponible
)
print("✅ Embeddings cargados")

# ========================
# CARGA DEL MODELO QA
# ========================
print(f"🤖 Cargando modelo de QA: {QA_MODEL}...")
qa_pipeline = pipeline(
    "question-answering", # Pipeline de preguntas y respuestas (extractivo)
    model=QA_MODEL,
    tokenizer=QA_MODEL
)
print("✅ Pipeline QA listo")

# ========================
# CARGA DEL ÍNDICE RAG
# ========================
# Lee el índice FAISS generado previamente y los metadatos (texto chunkificado)
print("📚 Cargando índice RAG...")
index = faiss.read_index("rag_index.faiss")
with open("rag_metadata.json", "r", encoding="utf-8") as f:
    metadatos_json = json.load(f)
chunks = metadatos_json.get("chunks", [])
print(f"✅ Índice y metadatos cargados ({len(chunks)} chunks)")

# ========================
# FUNCIONES RAG
# ========================
def recuperar_contexto(pregunta, k=3):
    """Recupera los k chunks más relevantes usando embeddings + FAISS"""
    # Convierte la pregunta en vector de embeddings
    pregunta_emb = embedding_model.encode(pregunta)
     # Busca los k más cercanos en el índice FAISS
    _, indices = index.search(np.array([pregunta_emb]), k)
    # Recupera el texto original de cada índice encontrado
    contextos = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
    return "\n\n".join(contextos)

def responder_pregunta(pregunta):
    """
    Usa el pipeline de QA (extractivo) para responder
    """
    contexto = recuperar_contexto(pregunta)
    if not contexto.strip():
        return "⚠️ No encontré contexto relevante en los documentos."
    # Pasa la pregunta y el contexto al modelo de QA extractivo
    resultado = qa_pipeline({
        "question": pregunta,
        "context": contexto
    })

    return resultado["answer"]

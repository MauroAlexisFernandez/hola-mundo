import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader  

# ============================
# CONFIG
# ============================
# Rutas y parámetros principales
PDF_PATH = "TFM_Mauro_Alexis_Fernandez.pdf" # PDF origen para construir el índice
CHUNK_SIZE = 700 # Cantidad de caracteres por fragmento (chunk)
CHUNK_OVERLAP = 100 # Superposición entre fragmentos para no perder contexto
INDEX_NAME = "rag_index.faiss" # Nombre del archivo donde se guardará el índice FAISS
METADATA_NAME = "rag_metadata.json"  # Archivo para guardar texto de cada chunk
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Modelo de embeddings

# ============================
# FUNCIONES
# ============================
def smart_chunk(text, chunk_size=700, overlap=100):
    """Divide un texto largo en chunks con solapamiento"""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        # Genera el fragmento desde start hasta start+chunk_size
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        # Avanza al siguiente fragmento restando el solapamiento
        start += chunk_size - overlap
    return chunks

def extract_chunks_from_pdf(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
         # Concatena texto de cada página
        full_text += page.extract_text() + "\n"
    return smart_chunk(full_text)

# ============================
# MAIN
# ============================
print("✅ Cargando modelo de embeddings...")
model = SentenceTransformer(MODEL_NAME) # Carga el modelo de transformadores para embeddings

print("📄 Extrayendo y dividiendo texto del PDF...")
chunks = extract_chunks_from_pdf(PDF_PATH) # Lee y chunkifica el PDF
print(f"✂️ Total de chunks generados desde el PDF: {len(chunks)}")

print("🔢 Generando embeddings...")
embeddings = model.encode(chunks, show_progress_bar=True) # Convierte cada chunk a un vector


print("💾 Guardando índice FAISS y metadatos...")
d = embeddings.shape[1]  # Dimensión de los vectores
index = faiss.IndexFlatL2(d) # Crea índice plano (L2)
index.add(np.array(embeddings))  # Agrega todos los vectores al índice
faiss.write_index(index, INDEX_NAME) # Guarda índice en disco

# Guarda también el texto original de cada chunk para referencia
with open(METADATA_NAME, "w", encoding="utf-8") as f:
    json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)

print("✅ ¡Índice RAG creado con éxito desde el PDF!")

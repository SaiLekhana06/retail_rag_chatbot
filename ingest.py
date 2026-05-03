"""
ingest.py — Load, Chunk, Embed, and Store product data into FAISS vector store.
Run this script ONCE before launching the chatbot.
Usage: python ingest.py
"""
 
import json
import csv
import os
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
 
# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "products.index")
CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "chunks.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, free, runs locally
 
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
 
 
def load_json_products(filepath):
    """Load products from products.json and convert each to a text chunk."""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        products = json.load(f)
    for p in products:
        chunk = (
            f"Product: {p['name']} (ID: {p['product_id']})\n"
            f"Category: {p['category']} | Brand: {p['brand']} | Price: {p['price']} | Stock: {p['stock_status']}\n"
            f"Description: {p['description']}\n"
            f"Features: {p['features']}\n"
            f"Warranty: {p['warranty']}\n"
            f"Return Policy: {p['return_policy']}"
        )
        chunks.append({"text": chunk, "source": "products.json", "product_id": p["product_id"], "name": p["name"]})
    return chunks
 
 
def load_txt_products(filepath):
    """Load products from products.txt (one blank-line-separated block per product)."""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # Split on double newlines to get individual product blocks
    blocks = [b.strip() for b in content.split("\n\n") if b.strip() and b.strip().startswith("Product:")]
    for block in blocks:
        name_line = block.split("\n")[0].replace("Product:", "").strip()
        chunks.append({"text": block, "source": "products.txt", "product_id": "TXT", "name": name_line})
    return chunks
 
 
def load_csv_products(filepath):
    """Load products from products.csv."""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunk = (
                f"Product: {row['name']} (ID: {row['product_id']})\n"
                f"Category: {row['category']} | Brand: {row['brand']} | Price: {row['price']} | Stock: {row['stock_status']}\n"
                f"Description: {row['description']}\n"
                f"Features: {row['features']}\n"
                f"Warranty: {row['warranty']}\n"
                f"Return Policy: {row['return_policy']}"
            )
            chunks.append({"text": chunk, "source": "products.csv", "product_id": row["product_id"], "name": row["name"]})
    return chunks
 
 
def embed_chunks(chunks, model):
    """Generate embeddings for all text chunks using SentenceTransformer."""
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with model '{EMBEDDING_MODEL}'...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype(np.float32)
 
 
def build_faiss_index(embeddings):
    """Build a FAISS IndexFlatL2 from the embedding matrix."""
    dim = embeddings.shape[1]   # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors of dimension {dim}.")
    return index
 
 
def main():
    print("=== Retail RAG Chatbot — Ingestion Pipeline ===")
 
    # 1. Load all product data
    all_chunks = []
    json_path = os.path.join(DATA_DIR, "products.json")
    txt_path  = os.path.join(DATA_DIR, "products.txt")
    csv_path  = os.path.join(DATA_DIR, "products.csv")
 
    if os.path.exists(json_path):
        j = load_json_products(json_path)
        all_chunks.extend(j)
        print(f"Loaded {len(j)} chunks from JSON.")
 
    if os.path.exists(txt_path):
        t = load_txt_products(txt_path)
        all_chunks.extend(t)
        print(f"Loaded {len(t)} chunks from TXT.")
 
    if os.path.exists(csv_path):
        c = load_csv_products(csv_path)
        all_chunks.extend(c)
        print(f"Loaded {len(c)} chunks from CSV.")
 
    if not all_chunks:
        print("ERROR: No product data found. Add files to the data/ folder.")
        return
 
    print(f"Total chunks to embed: {len(all_chunks)}")
 
    # 2. Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
 
    # 3. Generate embeddings
    embeddings = embed_chunks(all_chunks, model)
 
    # 4. Build FAISS index
    index = build_faiss_index(embeddings)
 
    # 5. Save index and chunks to disk
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
 
    print(f"Vector store saved to '{VECTOR_STORE_DIR}/'.")
    print("Ingestion complete! You can now run: streamlit run app.py")
 
 
if __name__ == "__main__":
    main()

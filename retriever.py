"""
retriever.py — Load FAISS index and retrieve top-K relevant product chunks
for a given user query using semantic similarity search.
"""
 
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
 
# ── Configuration ──────────────────────────────────────────────────────────
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "products.index")
CHUNKS_PATH      = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
TOP_K            = 3   # Number of relevant chunks to retrieve per query
 
 
class Retriever:
    """
    Loads the FAISS index once and exposes a retrieve() method.
    Designed to be instantiated once in app.py using st.cache_resource.
    """
 
    def __init__(self):
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError(
                "Vector store not found. Please run `python ingest.py` first."
            )
 
        print("Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
 
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
 
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
 
        print(f"Retriever ready. {self.index.ntotal} vectors loaded.")
 
    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Embed the query and retrieve the top_k most similar product chunks.
 
        Args:
            query: The user's natural language question.
            top_k: Number of chunks to return.
 
        Returns:
            List of chunk dicts with keys: text, source, product_id, name.
        """
        if not query.strip():
            return []
 
        # 1. Embed the query
        query_vector = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
 
        # 2. Similarity search in FAISS (L2 distance)
        distances, indices = self.index.search(query_vector, top_k)
 
        # 3. Collect and return matching chunks
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for empty slots
                continue
            chunk = self.chunks[idx].copy()
            chunk["distance"] = float(dist)
            results.append(chunk)
 
        return results

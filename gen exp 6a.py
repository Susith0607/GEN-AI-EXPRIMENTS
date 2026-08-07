# PROGRAM 1: CREATE KNOWLEDGE BASE & EMBEDDINGS

!pip install sentence-transformers faiss-cpu -q

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Step 1: Knowledge base
documents = [
"Generative AI creates content like text, images, and audio.",
"Large Language Models are used for text generation and chatbots.",
"RAG combines retrieval with text generation.",
"Vector databases store embeddings for similarity search.",
"Prompt engineering improves AI responses.",
"Fine-tuning adapts models to specific tasks."
]

# Step 2: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Convert to embeddings
embeddings = model.encode(documents, convert_to_numpy=True).astype("float32")

# Step 4: Normalize
faiss.normalize_L2(embeddings)

# Step 5: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("Knowledge Base Created Successfully!")
print("Total Documents:", len(documents))
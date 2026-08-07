# PROGRAM 2: RETRIEVE RELEVANT DOCUMENTS

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Same data
documents = [
"Generative AI creates content like text, images, and audio.",
"Large Language Models are used for text generation and chatbots.",
"RAG combines retrieval with text generation.",
"Vector databases store embeddings for similarity search.",
"Prompt engineering improves AI responses.",
"Fine-tuning adapts models to specific tasks."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Retrieval function
def retrieve(query, top_k=2):
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for i, s in zip(indices[0], scores[0]):
        results.append((documents[i], float(s)))
    return results

# Test
query = "What is RAG?"
results = retrieve(query)

print("Query:", query)
print("\nRetrieved Documents:")
for doc, score in results:
    print("-", doc)
    print("Score:", round(score, 4))
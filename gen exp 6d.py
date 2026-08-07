# PROGRAM 4: COMPLETE RAG SYSTEM

!pip install transformers sentence-transformers faiss-cpu torch -q

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Knowledge base
documents = [
"Generative AI creates content like text, images, and audio.",
"Large Language Models are used for text generation and chatbots.",
"RAG combines retrieval with text generation.",
"Vector databases store embeddings for similarity search.",
"Prompt engineering improves AI responses.",
"Fine-tuning adapts models to specific tasks."
]

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(doc_embeddings)

# FAISS index
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Generator model
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Retrieve
def retrieve(query):
    q_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, 2)

    return [documents[i] for i in indices[0]]

# Generate
def generate(query, docs):
    context = "\n".join(docs)
    prompt = f"""
Context:
{context}

Question:
{query}

Answer clearly:
"""
    return generator(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]

# Run
query = input("Enter your question: ")

docs = retrieve(query)
answer = generate(query, docs)

print("\nRetrieved Documents:")
for d in docs:
    print("-", d)

print("\nFinal Answer:")
print(answer)
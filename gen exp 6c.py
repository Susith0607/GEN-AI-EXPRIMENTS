# PROGRAM 3: GENERATE ANSWER FROM CONTEXT

!pip install transformers torch -q

from transformers import pipeline

generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base"
)

def generate_answer(query, context):
    prompt = f"""
Context:
{context}

Question:
{query}

Answer clearly using only context.
"""
    result = generator(prompt, max_new_tokens=100, do_sample=False)
    return result[0]["generated_text"]

# Example
context = "RAG combines retrieval with text generation."
query = "What is RAG?"

print("Generated Answer:")
print(generate_answer(query, context))
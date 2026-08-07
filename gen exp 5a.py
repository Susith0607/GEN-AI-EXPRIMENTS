"""
PROGRAM 1: CONTENT GENERATION USING FLAN-T5

INSTRUCTION:
- Generate a simple introduction to Generative AI
- Audience: First-year engineering students
- Use simple language
- Limit to 5 sentences
- Include 2 real-world applications
"""

from transformers import pipeline

# Load model
generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base"
)

def generate_response(prompt):
    result = generator(prompt, max_new_tokens=200, do_sample=False)
    return result[0]["generated_text"]

# Prompt
prompt = """
Role: You are an experienced Artificial Intelligence teacher.
Task: Write a simple introduction to Generative Artificial Intelligence.
Target audience: First-year engineering students.
Requirements:
1. Use simple language.
2. Limit the response to five sentences.
3. Include two real-world applications.
4. Avoid highly technical terms.
"""

# Output
print("=== CONTENT GENERATION ===")
print(generate_response(prompt))
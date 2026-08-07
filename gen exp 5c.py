"""
PROGRAM 3: EMAIL AUTOMATION

INSTRUCTION:
- Generate a formal email
- Include subject
- Mention time and venue
- Give clear instructions
"""

from transformers import pipeline

generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base"
)

def generate_response(prompt):
    result = generator(prompt, max_new_tokens=200, do_sample=False)
    return result[0]["generated_text"]

prompt = """
Role: You are a professional academic coordinator.
Task: Write a formal email to students.

Context:
A Generative AI lab session is scheduled for Friday at 10:00 AM 
in AI Laboratory 2. Students must bring laptops and complete 
Hugging Face registration.

Requirements:
1. Include subject
2. Professional tone
3. Mention time and venue
4. Clear instructions
5. Keep concise
"""

print("=== EMAIL GENERATION ===")
print(generate_response(prompt))
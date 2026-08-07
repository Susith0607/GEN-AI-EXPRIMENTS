"""
PROGRAM 2: REASONING TASK

INSTRUCTION:
- Solve a student completion problem
- Show steps clearly
- Provide final answer
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
Solve the following problem.
A college conducted a Generative AI workshop for 120 students.
Eighty-five students completed the workshop successfully.

Instructions:
1. Identify total students
2. Identify completed students
3. Calculate not completed
4. Provide explanation and final answer
"""

print("=== REASONING TASK ===")
print(generate_response(prompt))
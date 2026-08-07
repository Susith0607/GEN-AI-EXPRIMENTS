"""
PROGRAM 4: ACTION ITEM EXTRACTION

INSTRUCTION:
- Extract tasks from meeting notes
- Show:
  1. Person
  2. Task
  3. Deadline
- Output as numbered list
"""

from transformers import pipeline

generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base"
)

def generate_response(prompt):
    result = generator(prompt, max_new_tokens=200, do_sample=False)
    return result[0]["generated_text"]

meeting_notes = """
The project team reviewed the development of the college chatbot.
Arun will prepare the training dataset by Monday.
Priya will test the chatbot responses by Wednesday.
Rahul will prepare the final demonstration and presentation.
The team will meet again on Friday to review progress.
"""

prompt = f"""
Extract action items from the meeting notes.

Meeting Notes:
{meeting_notes}

For each action item provide:
1. Person responsible
2. Task
3. Deadline

Output as numbered list.
"""

print("=== ACTION ITEM EXTRACTION ===")
print(generate_response(prompt))
from transformers import pipeline

# Load zero-shot classification pipeline (explicit model recommended)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Input document
document = """
Artificial Intelligence and Machine Learning are transforming
industries through automation and intelligent decision-making.
"""

# Candidate labels
labels = ["Technology", "Sports", "Politics", "Entertainment"]

# Classify document
result = classifier(document, labels)

# Print full result
print(result)

# Print best label (clean output)
print("\nBest Category:", result["labels"][0])
print("Confidence Score:", result["scores"][0])
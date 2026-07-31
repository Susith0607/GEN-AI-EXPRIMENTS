from transformers import pipeline

# Load summarization pipeline with model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# Input text
text = """
Artificial Intelligence is transforming many industries by enabling
machines to perform tasks that normally require human intelligence.
It is widely used in healthcare, education, manufacturing, finance,
transportation, and cybersecurity. AI systems can analyze large
amounts of data, identify patterns, make predictions, and support
intelligent decision-making. Generative AI is a branch of Artificial
Intelligence that can create new content such as text, images, audio,
video, and computer programs.
"""

# Generate summary (with truncation safety)
result = summarizer(
    text,
    max_length=60,
    min_length=20,
    do_sample=False,
    truncation=True
)

# Display result
print("Summary:\n")
print(result[0]["summary_text"])
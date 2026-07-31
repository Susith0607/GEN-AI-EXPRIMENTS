from transformers import pipeline

# Load sentiment analysis pipeline (explicit model for stability)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Input text
text = "The Generative AI workshop was extremely informative and useful."

# Predict sentiment
result = sentiment_analyzer(text)

# Print output
print(result)
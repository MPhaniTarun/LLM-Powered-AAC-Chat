from flask import Flask, render_template, request, jsonify
import nltk
from nltk.util import ngrams
from collections import Counter

app = Flask(__name__)

# Sample text corpus for predictions
corpus = """
Hello, how are you? I am doing well. Thank you for your help. I would like some water.
Can you help me with this? I am hungry. Let's go to the park. I am feeling happy today.
"""

# Tokenize and generate n-grams
tokens = nltk.word_tokenize(corpus.lower())
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

# Build word prediction dictionaries
bigram_model = Counter(bigrams)
trigram_model = Counter(trigrams)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").lower().strip()
    words = text.split()

    predictions = set()

    if len(words) >= 2:
        last_two = tuple(words[-2:])
        for (w1, w2, w3) in trigram_model:
            if (w1, w2) == last_two:
                predictions.add(f"{text} {w3}")  # Suggest full sentence with prediction

    elif len(words) == 1:
        last_word = words[-1]
        for (w1, w2) in bigram_model:
            if w1 == last_word:
                predictions.add(f"{text} {w2}")  # Suggest next word

    return jsonify({"predictions": list(predictions)})

if __name__ == "__main__":
    app.run(debug=True)

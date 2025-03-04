from flask import Flask, render_template, request, jsonify
from gtts import gTTS
import os
import nltk
from nltk.util import ngrams
from collections import Counter

app = Flask(__name__)

# Ensure the static directory exists
os.makedirs("static", exist_ok=True)

AUDIO_FILE = "static/speech.mp3"  # Single filename to be reused

# Sample text corpus for word prediction
corpus = """
Hello, how are you? I am doing well. Thank you for your help. I would like some water.
Can you help me with this? I am hungry. Let's go to the park. I am feeling happy today.
My name is Tarun. I am doing great. Tarun is a great kid!.
"""

# Tokenize corpus
tokens = nltk.word_tokenize(corpus.lower())
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

# Build word prediction dictionaries
bigram_model = Counter(bigrams)
trigram_model = Counter(trigrams)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "")

    if text:
        tts = gTTS(text=text, lang="en")
        tts.save(AUDIO_FILE)  # Overwrite the same file
        return jsonify({"status": "success", "audio": AUDIO_FILE + "?t=" + str(os.path.getmtime(AUDIO_FILE))})

    return jsonify({"status": "error", "message": "No text provided"}), 400

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").lower()
    words = text.split()

    predictions = set()

    if len(words) >= 2:
        last_two = tuple(words[-2:])
        for (w1, w2, w3) in trigram_model:
            if (w1, w2) == last_two:
                predictions.add(w3)

    elif len(words) == 1:
        last_word = words[-1]
        for (w1, w2) in bigram_model:
            if w1 == last_word:
                predictions.add(w2)

    return jsonify({"predictions": list(predictions)})

if __name__ == "__main__":
    app.run(debug=True)

# pip install -r requirements.txt
# python app.py
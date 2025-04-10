from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import gensim.downloader as api
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained Word2Vec model (limited to 50,000 words for efficiency)
model_path = api.load("word2vec-google-news-300", return_path=True)
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=50000)

# Load dataset
try:
    df = pd.read_csv("processed_dataset.csv")
    if "Desired_answer" not in df.columns or "Question" not in df.columns:
        raise KeyError("Required columns ('Desired_answer', 'Question') missing in dataset!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.DataFrame(columns=["Question", "Desired_answer"])  # Fallback empty DataFrame

# Load trained LSTM model & tokenizer
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}  # Fix MSE loss loading
model = tf.keras.models.load_model("lstm_model.h5", custom_objects=custom_objects)

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define max sequence length
MAX_LENGTH = 50

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    """Tokenizes, removes stopwords, and lemmatizes the input text."""
    if not isinstance(text, str) or text.strip() == "":
        return ""

    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]

    return " ".join(tokens)

# Function to compute Word Mover's Distance (WMD)
def compute_wmd(text1, text2):
    """Computes the WMD between two texts using Word2Vec embeddings."""
    text1_tokens = text1.split()
    text2_tokens = text2.split()

    if not text1_tokens or not text2_tokens:  # Avoid empty token lists
        return float("inf")

    try:
        wmd = word_vectors.wmdistance(text1_tokens, text2_tokens)
        return wmd if wmd != float("inf") else 2  # Handle infinity cases
    except Exception:
        return 2  # Default high distance if error occurs

# Function to compute Cosine Similarity
vectorizer = TfidfVectorizer()
if not df.empty:
    vectorizer.fit(df["Desired_answer"].dropna().tolist())

def compute_cosine_similarity(text1, text2):
    """Computes the cosine similarity between two texts."""
    if not text1 or not text2:
        return 0.0  # Avoid errors with empty texts

    try:
        text_vectors = vectorizer.transform([text1, text2])
        return cosine_similarity(text_vectors[0], text_vectors[1])[0][0]
    except Exception:
        return 0.0  # Default low similarity if an error occurs

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_question", methods=["GET"])
def get_question():
    """Fetch a random question from the dataset."""
    if df.empty:
        return jsonify({"error": "No questions available!"}), 500

    random_row = df.sample(n=1).iloc[0]
    question = random_row["Question"]
    correct_answer = random_row["Desired_answer"]
    return jsonify({"question": question, "correct_answer": correct_answer})

@app.route("/predict", methods=["POST"])
def predict():
    """Processes user's answer, computes similarity, and predicts score."""
    user_name = request.form.get("name")
    user_answer = request.form.get("answer")
    correct_answer = request.form.get("correct_answer")

    if not user_answer or not correct_answer:
        return jsonify({"error": "Invalid input. Both question and answer are required!"}), 400

    # Preprocess answers
    user_answer_processed = preprocess_text(user_answer)
    correct_answer_processed = preprocess_text(correct_answer)

    if not user_answer_processed or not correct_answer_processed:
        return jsonify({"error": "Processed answers are empty!"}), 400

    # Compute similarity scores
    wmd_score = compute_wmd(user_answer_processed, correct_answer_processed)
    cosine_sim = compute_cosine_similarity(user_answer_processed, correct_answer_processed)

    # Normalize similarity scores (higher is better)
    similarity_score = (1 - min(wmd_score, 1)) * 5  # WMD is distance-based
    similarity_score = max(0, min(5, similarity_score))  # Clamp to [0, 5]

    # Convert text to sequence for LSTM model
    seq = tokenizer.texts_to_sequences([user_answer_processed])
    padded_seq = pad_sequences(seq, maxlen=MAX_LENGTH)

    # Predict score using trained LSTM model
    predicted_score = model.predict(padded_seq)[0][0]

    # Weighted combination of LSTM prediction and similarity score
    final_score = (0.7 * predicted_score) + (0.3 * similarity_score)
    final_score = max(0, min(5, final_score))  # Clamp to [0, 5]

    return jsonify({
        "name": user_name,
        "score": round(final_score, 2),
        "cosine_similarity": round(cosine_sim, 2),
        "wmd_score": round(wmd_score, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import pandas as pd
import random

app = Flask(__name__)

# Load dataset and ensure required columns exist
try:
    df = pd.read_csv('../train/processed_dataset.csv')
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    required_columns = {"question", "desired_answer"}

    if not required_columns.issubset(df.columns):
        raise KeyError(f"Required columns {required_columns} are missing!")

except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.DataFrame(columns=["question", "desired_answer"])  # Empty DataFrame fallback


def get_random_question():
    """Get a random question and its desired answer from the dataset."""
    if df.empty:
        return "No questions available", "No answers available"
    
    random_row = df.sample(n=1).iloc[0]
    return random_row['question'], random_row['desired_answer']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Handles the demo page for evaluating answers."""
    question, desired_answer = get_random_question()
    result = None

    if request.method == 'POST':
        student_answer = request.form.get('student_answer')
        similarity_score = evaluate_answer(desired_answer, student_answer)

        result = {
            'score': similarity_score,
            'feedback': generate_feedback(similarity_score)
        }

    return render_template('demo.html', question=question, desired_answer=desired_answer, result=result)


def evaluate_answer(desired_answer, student_answer):
    """Evaluates student's answer using a simple word overlap technique."""
    if not desired_answer or not student_answer:
        return 0

    common_words = set(desired_answer.lower().split()) & set(student_answer.lower().split())
    total_words = set(desired_answer.lower().split())

    return round((len(common_words) / len(total_words)) * 100) if total_words else 0


def generate_feedback(score):
    """Generates feedback based on the similarity score."""
    if score >= 90:
        return "Excellent! Your answer closely matches the expected response."
    elif score >= 70:
        return "Good job! Your answer covers most of the key points."
    elif score >= 50:
        return "Satisfactory. Your answer includes some key points but misses others."
    elif score >= 30:
        return "Needs improvement. Your answer is missing many key points."
    else:
        return "Your answer differs significantly from the expected response. Please review the material."


if __name__ == '_main_':
    app.run(debug=True)
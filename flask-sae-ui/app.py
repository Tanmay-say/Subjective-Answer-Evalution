from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    result = None
    if request.method == 'POST':
        question = request.form.get('question')
        desired_answer = request.form.get('desired_answer')
        student_answer = request.form.get('student_answer')
        
        # This is a placeholder for your actual evaluation logic
        # In a real implementation, you would call your evaluation model here
        similarity_score = evaluate_answer(question, desired_answer, student_answer)
        
        result = {
            'score': similarity_score,
            'feedback': generate_feedback(similarity_score)
        }
    
    return render_template('demo.html', result=result)

def evaluate_answer(question, desired_answer, student_answer):
    """
    Placeholder function for answer evaluation.
    In a real implementation, this would use your NLP model.
    """
    # Simple placeholder logic - in reality, you'd use your ML model here
    if not question or not desired_answer or not student_answer:
        return 0
    
    # Very basic similarity check (for demonstration only)
    common_words = set(desired_answer.lower().split()) & set(student_answer.lower().split())
    total_words = set(desired_answer.lower().split())
    
    if not total_words:
        return 0
    
    return round((len(common_words) / len(total_words)) * 100)

def generate_feedback(score):
    """Generate feedback based on the similarity score."""
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

if __name__ == '__main__':
    app.run(debug=True)


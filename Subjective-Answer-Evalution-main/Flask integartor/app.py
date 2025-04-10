import os
import re
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



# Disable GPU if not needed
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# Constants from notebook
MAX_QUESTION_LEN = 100
MAX_DESIRED_ANSWER_LEN = 300
MAX_STUDENT_ANSWER_LEN = 300

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'subjective_answer_evaluation.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'answer_tokenizer.pickle')
SCALER_PATH = os.path.join(BASE_DIR, 'length_features_scaler.pickle')

# Initialize NLP components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and normalize text data"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def create_prediction_pipeline(model_path, tokenizer_path, scaler_path):
    """Create prediction pipeline with loaded resources"""
    try:
        print("\n=== Initializing Model Pipeline ===")
        print(f"Current directory: {BASE_DIR}")
        print(f"Directory contents: {os.listdir(BASE_DIR)}")
        
        print("\nChecking for model files:")
        print(f"Model path: {model_path} - Exists: {os.path.exists(model_path)}")
        print(f"Tokenizer path: {tokenizer_path} - Exists: {os.path.exists(tokenizer_path)}")
        print(f"Scaler path: {scaler_path} - Exists: {os.path.exists(scaler_path)}")
        
        if not all(os.path.exists(p) for p in [model_path, tokenizer_path, scaler_path]):
            missing = [p for p in [model_path, tokenizer_path, scaler_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"Missing files: {missing}")
            
        print("\nLoading model...")
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        print(f"Model type: {type(model)}")
        
        print("Loading tokenizer...")
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"Tokenizer type: {type(tokenizer)}")
        
        print("Loading scaler...")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler type: {type(scaler)}")
        
        return model, tokenizer, scaler
        
    except Exception as e:
        print(f"\n!!! Initialization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Initialize pipeline with debug info
print("\n=== Starting Initialization ===")
model, tokenizer, scaler = create_prediction_pipeline(
    MODEL_PATH,
    TOKENIZER_PATH,
    SCALER_PATH
)

def predict_score(question, desired_answer, student_answer):
    """Main prediction function"""
    try:
        # Preprocess texts
        question_proc = preprocess_text(question)
        desired_proc = preprocess_text(desired_answer)
        student_proc = preprocess_text(student_answer)
        
        # Tokenize and pad sequences
        question_seq = tokenizer.texts_to_sequences([question_proc])
        desired_seq = tokenizer.texts_to_sequences([desired_proc])
        student_seq = tokenizer.texts_to_sequences([student_proc])
        
        question_padded = tf.keras.preprocessing.sequence.pad_sequences(
            question_seq, maxlen=MAX_QUESTION_LEN, padding='post')
        desired_padded = tf.keras.preprocessing.sequence.pad_sequences(
            desired_seq, maxlen=MAX_DESIRED_ANSWER_LEN, padding='post')
        student_padded = tf.keras.preprocessing.sequence.pad_sequences(
            student_seq, maxlen=MAX_STUDENT_ANSWER_LEN, padding='post')
            
        # Create length features
        q_len = len(question_proc)
        d_len = len(desired_proc)
        s_len = len(student_proc)
        len_ratio = s_len / (d_len + 1e-7)  # Prevent division by zero
        
        length_features = scaler.transform([[q_len, d_len, s_len, len_ratio]])
        
        # Make prediction
        prediction = model.predict([
            question_padded,
            desired_padded,
            student_padded,
            length_features
        ], verbose=0)
        
        # Clip and round final score
        final_score = np.clip(prediction[0][0], 0, 5)
        return round(final_score, 1)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    grade = None
    error = None
    
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        desired_answer = request.form.get('desired_answer', '').strip()
        student_answer = request.form.get('student_answer', '').strip()

        if all([question, desired_answer, student_answer]):
            if model and tokenizer and scaler:
                try:
                    grade = predict_score(question, desired_answer, student_answer)
                except Exception as e:
                    error = f"Prediction error: {str(e)}"
            else:
                error = "Model initialization failed - check server logs"
        else:
            error = "Please fill all fields"
    
    return render_template('index.html', grade=grade, error=error)

@app.route('/api/grade', methods=['POST'])
def api_grade():
    data = request.json
    required_fields = ['question', 'desired_answer', 'student_answer']
    
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
        
    if not all(data[field].strip() for field in required_fields):
        return jsonify({'error': 'Empty fields detected'}), 400
        
    if model and tokenizer and scaler:
        try:
            grade = predict_score(
                data['question'].strip(),
                data['desired_answer'].strip(),
                data['student_answer'].strip()
            )
            return jsonify({'grade': grade})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Model initialization failed'}), 500

if __name__ == '__main__':
    print("\n=== Starting Flask Application ===")
    app.run(debug=True)
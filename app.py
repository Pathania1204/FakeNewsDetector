import os
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import joblib
import pandas as pd
from model_utils import preprocess_text, load_model_and_vectorizer
from werkzeug.utils import secure_filename
import csv

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fake-news-detector-secret-key")

# Configuration
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv', 'txt'}
MODEL_FOLDER = 'model'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with news input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if news is fake or real"""
    try:
        news_text = request.form.get('news_text', '').strip()
        
        if not news_text:
            flash('Please enter some news text to analyze.', 'error')
            return redirect(url_for('index'))
        
        # Load model and vectorizer
        model, vectorizer = load_model_and_vectorizer()
        
        if model is None or vectorizer is None:
            flash('Model not found. Please train the model first.', 'error')
            return redirect(url_for('index'))
        
        # Preprocess and predict
        processed_text = preprocess_text(news_text)
        text_vectorized = vectorizer.transform([processed_text])
        
        # Get probability prediction
        probability = model.predict_proba(text_vectorized)[0]
        fake_probability = probability[0] * 100  # Assuming 0 is fake, 1 is real
        real_probability = probability[1] * 100
        
        # Determine prediction
        prediction = "FAKE" if fake_probability > real_probability else "REAL"
        confidence = max(fake_probability, real_probability)
        
        result = {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'fake_probability': round(fake_probability, 2),
            'real_probability': round(real_probability, 2),
            'news_text': news_text[:200] + "..." if len(news_text) > 200 else news_text
        }
        
        return render_template('index.html', result=result)
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        flash(f'An error occurred during prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Train the model with available dataset"""
    if request.method == 'POST':
        try:
            from train_model import train_and_save_model
            
            # Get dataset file
            dataset_file = request.form.get('dataset_file', 'data/sample_news.csv')
            
            if not os.path.exists(dataset_file):
                flash('Dataset file not found. Please upload a dataset first.', 'error')
                return redirect(url_for('train_model'))
            
            # Train model
            accuracy = train_and_save_model(dataset_file)
            flash(f'Model trained successfully! Accuracy: {accuracy:.2%}', 'success')
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            flash(f'Error during training: {str(e)}', 'error')
    
    # List available datasets
    datasets = []
    if os.path.exists(UPLOAD_FOLDER):
        for file in os.listdir(UPLOAD_FOLDER):
            if file.endswith('.csv'):
                datasets.append(file)
    
    return render_template('train.html', datasets=datasets)

@app.route('/upload', methods=['POST'])
def upload_dataset():
    """Upload a custom dataset"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('train_model'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('train_model'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Validate CSV format
        try:
            df = pd.read_csv(filepath)
            required_columns = ['text', 'label']
            
            if not all(col in df.columns for col in required_columns):
                os.remove(filepath)
                flash('Invalid CSV format. Please ensure columns "text" and "label" exist.', 'error')
                return redirect(url_for('train_model'))
            
            flash(f'Dataset uploaded successfully! File: {filename}', 'success')
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(f'Error validating dataset: {str(e)}', 'error')
    else:
        flash('Invalid file type. Please upload a CSV or TXT file.', 'error')
    
    return redirect(url_for('train_model'))

@app.route('/model_status')
def model_status():
    """Check if model is trained and ready"""
    model_path = os.path.join(MODEL_FOLDER, 'fake_news_model.pkl')
    vectorizer_path = os.path.join(MODEL_FOLDER, 'tfidf_vectorizer.pkl')
    
    model_exists = os.path.exists(model_path) and os.path.exists(vectorizer_path)
    
    return jsonify({
        'model_trained': model_exists,
        'model_path': model_path if model_exists else None,
        'vectorizer_path': vectorizer_path if model_exists else None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

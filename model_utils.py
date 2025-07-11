import re
import string
import nltk
import joblib
import os
import logging
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    stop_words = set()

def preprocess_text(text):
    """
    Preprocess text for machine learning model
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 2]
        
        # Join tokens back to string
        processed_text = ' '.join(tokens)
        
        return processed_text
        
    except Exception as e:
        logging.error(f"Error preprocessing text: {str(e)}")
        return text  # Return original text if preprocessing fails

def load_model_and_vectorizer():
    """
    Load the trained model and vectorizer
    
    Returns:
        tuple: (model, vectorizer) or (None, None) if not found
    """
    try:
        model_path = os.path.join('model', 'fake_news_model.pkl')
        vectorizer_path = os.path.join('model', 'tfidf_vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            logging.warning("Model or vectorizer not found")
            return None, None
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        logging.info("Model and vectorizer loaded successfully")
        return model, vectorizer
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

def validate_dataset(file_path):
    """
    Validate dataset format
    
    Args:
        file_path (str): Path to dataset file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        import pandas as pd
        
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_columns = ['text', 'label']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for empty data
        if df.empty or df['text'].isna().all() or df['label'].isna().all():
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating dataset: {str(e)}")
        return False

def get_model_info():
    """
    Get information about the trained model
    
    Returns:
        dict: Model information
    """
    try:
        model_path = os.path.join('model', 'fake_news_model.pkl')
        vectorizer_path = os.path.join('model', 'tfidf_vectorizer.pkl')
        
        info = {
            'model_exists': os.path.exists(model_path),
            'vectorizer_exists': os.path.exists(vectorizer_path),
            'model_size': None,
            'vectorizer_size': None,
            'model_modified': None,
            'vectorizer_modified': None
        }
        
        if info['model_exists']:
            stat = os.stat(model_path)
            info['model_size'] = stat.st_size
            info['model_modified'] = stat.st_mtime
        
        if info['vectorizer_exists']:
            stat = os.stat(vectorizer_path)
            info['vectorizer_size'] = stat.st_size
            info['vectorizer_modified'] = stat.st_mtime
        
        return info
        
    except Exception as e:
        logging.error(f"Error getting model info: {str(e)}")
        return {'error': str(e)}

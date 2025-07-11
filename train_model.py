import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import logging
from model_utils import preprocess_text

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def load_dataset(file_path):
    """Load dataset from CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['text', 'label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")
        
        # Clean the data
        df = df.dropna(subset=['text', 'label'])
        df['text'] = df['text'].astype(str)
        
        # Standardize labels (convert to binary: 0 for fake, 1 for real)
        label_mapping = {
            'fake': 0, 'FAKE': 0, 'False': 0, 'false': 0, 0: 0, '0': 0,
            'real': 1, 'REAL': 1, 'True': 1, 'true': 1, 1: 1, '1': 1,
            'reliable': 1, 'unreliable': 0
        }
        
        df['label'] = df['label'].map(label_mapping)
        df = df.dropna(subset=['label'])  # Remove unmapped labels
        
        logging.info(f"Dataset loaded: {len(df)} samples")
        logging.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def train_and_save_model(dataset_path):
    """Train the fake news detection model and save it"""
    try:
        # Load dataset
        df = load_dataset(dataset_path)
        
        # Preprocess text data
        logging.info("Preprocessing text data...")
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Split data
        X = df['processed_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create TF-IDF vectorizer
        logging.info("Creating TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        
        # Fit and transform training data
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train Logistic Regression model
        logging.info("Training Logistic Regression model...")
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Model accuracy: {accuracy:.4f}")
        logging.info("\nClassification Report:")
        logging.info(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        # Save model and vectorizer
        model_dir = 'model'
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'fake_news_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logging.info(f"Model saved to: {model_path}")
        logging.info(f"Vectorizer saved to: {vectorizer_path}")
        
        return accuracy
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

def create_sample_dataset():
    """Create a sample dataset for testing"""
    sample_data = [
        # Fake news examples - more diverse and realistic
        {
            'text': 'BREAKING: Scientists discover that drinking water causes cancer in 99% of cases, government covers up the truth!',
            'label': 'fake'
        },
        {
            'text': 'Shocking revelation: Vaccines contain microchips designed to control your mind, leaked documents prove conspiracy',
            'label': 'fake'
        },
        {
            'text': 'Celebrity dies in tragic accident, family confirms the news was fabricated for publicity stunt',
            'label': 'fake'
        },
        {
            'text': 'New study shows that eating pizza every day can make you live forever, doctors hate this one simple trick',
            'label': 'fake'
        },
        {
            'text': 'Government announces mandatory curfew starting tomorrow, all citizens must stay indoors or face imprisonment',
            'label': 'fake'
        },
        {
            'text': 'Alien invasion confirmed by military sources, UFOs spotted over major cities worldwide',
            'label': 'fake'
        },
        {
            'text': 'Miracle cure discovered: This common household item can cure all diseases instantly',
            'label': 'fake'
        },
        {
            'text': 'President secretly replaced by body double, insider reveals shocking truth',
            'label': 'fake'
        },
        {
            'text': 'Scientists confirm flat earth theory, NASA admits decades of deception',
            'label': 'fake'
        },
        {
            'text': 'New law requires all citizens to surrender their pets to government facilities',
            'label': 'fake'
        },
        {
            'text': 'EXCLUSIVE: Hollywood actor found dead in hotel room, police suspect murder by rival celebrity',
            'label': 'fake'
        },
        {
            'text': 'Government secretly adding mind control chemicals to tap water supply across major cities',
            'label': 'fake'
        },
        {
            'text': 'Breaking: Internet will be shut down permanently next week due to cyberattack threat',
            'label': 'fake'
        },
        {
            'text': 'Scientists prove that 5G towers cause coronavirus spread, WHO refuses to comment',
            'label': 'fake'
        },
        {
            'text': 'Billionaire announces plan to buy all social media platforms and delete them forever',
            'label': 'fake'
        },
        {
            'text': 'URGENT: Meteor headed toward Earth, NASA covers up impending disaster',
            'label': 'fake'
        },
        {
            'text': 'New study reveals that smartphones are actually government surveillance devices',
            'label': 'fake'
        },
        {
            'text': 'Doctor discovers that sleeping 12 hours a day prevents aging completely',
            'label': 'fake'
        },
        {
            'text': 'Breaking: Time travel machine invented by teenager in garage, government seizes device',
            'label': 'fake'
        },
        {
            'text': 'Shocking: Popular food chain secretly using human meat in their burgers',
            'label': 'fake'
        },
        # Real news examples - more diverse and realistic
        {
            'text': 'The Federal Reserve announced a quarter-point interest rate increase following their latest meeting on monetary policy',
            'label': 'real'
        },
        {
            'text': 'Research published in Nature journal shows promising results for new cancer treatment in clinical trials',
            'label': 'real'
        },
        {
            'text': 'Local school district receives federal funding to improve technology infrastructure in public schools',
            'label': 'real'
        },
        {
            'text': 'Weather service issues tornado watch for three counties as severe thunderstorms approach the region',
            'label': 'real'
        },
        {
            'text': 'University researchers develop new method for recycling plastic waste into construction materials',
            'label': 'real'
        },
        {
            'text': 'Stock market closes higher as investors react positively to quarterly earnings reports',
            'label': 'real'
        },
        {
            'text': 'Public health officials recommend flu vaccination ahead of winter season',
            'label': 'real'
        },
        {
            'text': 'Construction begins on new highway bridge to improve traffic flow in metro area',
            'label': 'real'
        },
        {
            'text': 'University announces new scholarship program for students pursuing STEM degrees',
            'label': 'real'
        },
        {
            'text': 'Environmental agency releases annual report on air quality improvements in urban areas',
            'label': 'real'
        },
        {
            'text': 'Technology company reports strong quarterly earnings, stock price rises 5 percent',
            'label': 'real'
        },
        {
            'text': 'City council approves budget for new public library construction project',
            'label': 'real'
        },
        {
            'text': 'Medical researchers at Johns Hopkins publish study on diabetes prevention methods',
            'label': 'real'
        },
        {
            'text': 'National park service announces new conservation program to protect endangered species',
            'label': 'real'
        },
        {
            'text': 'Transportation department completes road repair project ahead of schedule',
            'label': 'real'
        },
        {
            'text': 'Local hospital receives accreditation for excellence in patient care standards',
            'label': 'real'
        },
        {
            'text': 'Energy company announces investment in renewable solar power infrastructure',
            'label': 'real'
        },
        {
            'text': 'Education ministry releases new curriculum guidelines for mathematics teaching',
            'label': 'real'
        },
        {
            'text': 'Fire department responds to apartment building fire, no injuries reported',
            'label': 'real'
        },
        {
            'text': 'Agricultural department issues guidelines for sustainable farming practices',
            'label': 'real'
        }
    ]
    
    all_data = sample_data
    
    # Save to CSV
    df = pd.DataFrame(all_data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_news.csv', index=False)
    
    logging.info(f"Sample dataset created with {len(all_data)} examples")
    return 'data/sample_news.csv'

if __name__ == '__main__':
    # Create sample dataset if it doesn't exist
    if not os.path.exists('data/sample_news.csv'):
        create_sample_dataset()
    
    # Train model with sample dataset
    accuracy = train_and_save_model('data/sample_news.csv')
    print(f"Model trained with accuracy: {accuracy:.2%}")

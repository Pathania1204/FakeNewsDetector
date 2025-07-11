# Fake News Detector

## Overview

This is a Flask-based web application that uses machine learning to detect fake news articles. The application allows users to input news text and receive predictions about whether the content is likely fake or real. It includes functionality for training custom models with user-uploaded datasets and uses a TF-IDF vectorizer with logistic regression for classification.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Bootstrap 5 with dark theme
- **JavaScript**: Vanilla JavaScript for UI interactions and AJAX calls
- **Template Engine**: Jinja2 (Flask's built-in templating)
- **Styling**: Custom CSS with Bootstrap components and Font Awesome icons

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Architecture Pattern**: MVC (Model-View-Controller)
- **File Structure**: Modular design with separate utilities and model training modules
- **Session Management**: Flask's built-in session management with secret key

### Machine Learning Pipeline
- **Text Processing**: NLTK for natural language processing
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classification**: Logistic Regression from scikit-learn
- **Model Persistence**: Joblib for saving/loading trained models

## Key Components

### Core Application Files
- `app.py`: Main Flask application with routes and request handling
- `model_utils.py`: Text preprocessing utilities and model loading functions
- `train_model.py`: Model training pipeline and dataset handling
- `templates/`: HTML templates for different pages (base, index, train)
- `static/`: CSS, JavaScript, and other static assets

### Data Processing
- **Text Preprocessing**: Removes URLs, mentions, hashtags, punctuation, and numbers
- **Tokenization**: NLTK word tokenization
- **Stemming**: Porter Stemmer for word normalization
- **Stopword Removal**: English stopwords filtering

### Model Training
- **Dataset Support**: CSV files with 'text' and 'label' columns
- **Label Standardization**: Converts various label formats to binary (0/1)
- **Train/Test Split**: Automated splitting for model evaluation
- **Model Evaluation**: Accuracy scoring and classification reports

## Data Flow

1. **User Input**: News text entered through web form
2. **Text Preprocessing**: Raw text cleaned and normalized
3. **Feature Extraction**: TF-IDF vectorization of preprocessed text
4. **Model Prediction**: Logistic regression classification
5. **Result Display**: Prediction with confidence score shown to user

### Training Flow
1. **Dataset Upload**: User uploads CSV file through web interface
2. **Data Validation**: Checks for required columns and data integrity
3. **Text Preprocessing**: Applies same preprocessing as prediction pipeline
4. **Model Training**: TF-IDF vectorizer and logistic regression training
5. **Model Persistence**: Saves trained model and vectorizer using joblib

## External Dependencies

### Python Libraries
- **Flask**: Web framework and routing
- **scikit-learn**: Machine learning algorithms and utilities
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **nltk**: Natural language processing
- **joblib**: Model serialization

### Frontend Dependencies
- **Bootstrap 5**: UI framework (loaded from CDN)
- **Font Awesome**: Icon library (loaded from CDN)
- **Custom CSS**: Application-specific styling

### NLTK Data
- **punkt**: Tokenizer data
- **stopwords**: English stopwords corpus
- **Automatic Download**: Downloads required data on first run

## Deployment Strategy

### File Organization
- **Static Files**: CSS, JavaScript, and assets in `static/` directory
- **Templates**: HTML templates in `templates/` directory
- **Data Storage**: Uploaded datasets in `data/` directory
- **Model Storage**: Trained models in `model/` directory

### Configuration
- **Environment Variables**: Session secret key configurable via environment
- **File Upload Limits**: 16MB maximum file size
- **Allowed File Types**: CSV and TXT files only
- **Directory Creation**: Automatic creation of required directories

### Security Considerations
- **File Upload Security**: Secure filename handling and type validation
- **Session Management**: Configurable secret key for session security
- **Input Validation**: Form validation and error handling
- **File Size Limits**: Prevents abuse through large file uploads

### Error Handling
- **Model Validation**: Checks for trained model availability
- **Dataset Validation**: Validates CSV structure and content
- **User Feedback**: Flash messages for user notifications
- **Logging**: Comprehensive logging for debugging and monitoring
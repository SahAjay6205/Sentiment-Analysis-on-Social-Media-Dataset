from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import re
import emoji
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from datetime import datetime

app = Flask(__name__)

# Load model and vectorizer
vectorizer = joblib.load('tfidf_vectorizer_[TIMESTAMP].pkl')  # Replace with your file
model = joblib.load('sentiment_model_[TIMESTAMP].pkl')        # Replace with your file

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def enhanced_clean_text(text):
    if pd.isna(text):
        return ""
    
    # Emoji handling
    emoji_map = {
        "👎": " negative_emoji ", "😠": " negative_emoji ",
        "😡": " negative_emoji ", "💩": " negative_emoji ",
        "👍": " positive_emoji ", "❤️": " positive_emoji ",
        "😍": " positive_emoji ", "😊": " positive_emoji "
    }
    for e, tag in emoji_map.items():
        text = str(text).replace(e, tag)
    
    # Text cleaning
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = contractions.fix(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    tokens = [word for word in tokens 
              if word not in stopwords.words('english') 
              and len(word) > 1]
    
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        cleaned = enhanced_clean_text(text)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][pred]
        
        result = {
            'text': text,
            'cleaned': cleaned,
            'sentiment': 'positive' if pred == 1 else 'negative',
            'confidence': float(proba),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
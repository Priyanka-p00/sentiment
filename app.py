import os
from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the model and vectorizer
def load_model_vectorizer():
    model_path = '/Users/prathvishmkumar/Documents/SentimentAnalysisApp/saved modelsentiment_model.pkl'
    vectorizer_path = '//Users/prathvishmkumar/Documents/SentimentAnalysisApp/saved modeltfidf_vectorizer.pkl'
    
    # Debug print statements to verify paths
    print("Model path:", model_path)
    print("Vectorizer path:", vectorizer_path)
    print("Model path exists:", os.path.exists(model_path))
    print("Vectorizer path exists:", os.path.exists(vectorizer_path))

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Initialize stop words for text cleaning
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()                  # Convert to lowercase
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    tweet = None
    if request.method == 'POST':
        tweet = request.form['tweet']  # Get tweet input from form
        model, vectorizer = load_model_vectorizer()  # Load model and vectorizer
        
        # Clean and vectorize the input tweet
        cleaned_tweet = clean_text(tweet)
        tweet_tfidf = vectorizer.transform([cleaned_tweet])  # Apply transform here to vectorizer
        
        # Predict sentiment using the loaded model
        sentiment = model.predict(tweet_tfidf)[0]  # Get the first (and only) prediction

    # Render the homepage with result
    return render_template('index.html', sentiment=sentiment, tweet=tweet)

if __name__ == '__main__':
    app.run(debug=True)

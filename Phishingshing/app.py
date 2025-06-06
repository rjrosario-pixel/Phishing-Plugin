from flask import Flask, request, jsonify, render_template
import re
from urllib.parse import urlparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# --- Feature extraction function ---
def extract_url_features(url):
    features = {}
    parsed = urlparse(url)
    domain = parsed.netloc
    features['url_length'] = len(url)
    features['count_dots'] = domain.count('.')
    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    features['has_ip'] = 1 if re.search(ip_pattern, domain) else 0
    features['count_at'] = url.count('@')
    features['count_hyphen'] = domain.count('-')
    features['count_question'] = url.count('?')
    features['count_equal'] = url.count('=')
    features['count_double_slash'] = url[8:].count('//')  # ignoring protocol
    features['https'] = 1 if parsed.scheme == 'https' else 0
    features['domain_length'] = len(domain)
    features['subdomain_count'] = domain.count('.') - 1 if domain.count('.') > 1 else 0
    suspicious_words = ['login', 'signin', 'bank', 'update', 'free', 'lucky', 'bonus', 'account']
    features['suspicious_words'] = int(any(word in url.lower() for word in suspicious_words))
    return features

# --- Dummy training data ---
# Replace this with your real dataset and training
def train_dummy_model():
    # Create dummy data: features + label (0=legit, 1=phishing)
    data = [
        # url_length, count_dots, has_ip, count_at, count_hyphen, count_question, count_equal, count_double_slash, https, domain_length, subdomain_count, suspicious_words, label
        [50, 2, 0, 0, 1, 0, 0, 0, 1, 15, 1, 0, 0],  # legit example
        [90, 4, 1, 1, 3, 2, 1, 1, 0, 25, 3, 1, 1],  # phishing example
        [45, 1, 0, 0, 0, 0, 0, 0, 1, 10, 0, 0, 0],  # legit
        [100, 5, 1, 2, 4, 3, 2, 2, 0, 30, 4, 1, 1]  # phishing
    ]
    df = pd.DataFrame(data, columns=[
        'url_length', 'count_dots', 'has_ip', 'count_at', 'count_hyphen', 'count_question', 'count_equal',
        'count_double_slash', 'https', 'domain_length', 'subdomain_count', 'suspicious_words', 'label'
    ])
    X = df.drop('label', axis=1)
    y = df['label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_dummy_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get('url', '')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    features = extract_url_features(url)
    # Convert features dict to DataFrame with correct column order
    feature_order = ['url_length', 'count_dots', 'has_ip', 'count_at', 'count_hyphen', 'count_question',
                     'count_equal', 'count_double_slash', 'https', 'domain_length', 'subdomain_count', 'suspicious_words']
    X = pd.DataFrame([features], columns=feature_order)
    
    prediction = model.predict(X)[0]
    label = 'Phishing' if prediction == 1 else 'Legitimate'
    
    return jsonify({'result': label})

if __name__ == '__main__':
    app.run(debug=True)

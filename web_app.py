#!/usr/bin/env python3
"""
SIMPLE WEB INTERFACE FOR SPAM DETECTOR
Run: python3 web_app.py then visit http://localhost:5000
"""

from flask import Flask, render_template_string, request
import joblib

# Load model
model_data = joblib.load('models/best_spam_model.joblib')
model = model_data['model']
vectorizer = model_data['vectorizer']

app = Flask(__name__)

# Simple HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Faiyaz's AI Spam Detector</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 30px; border-radius: 10px; }
        input, textarea { width: 100%; padding: 10px; margin: 10px 0; }
        button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 5px; }
        .spam { color: red; font-weight: bold; }
        .ham { color: green; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Faiyaz's AI Spam Detector</h1>
        <form method="POST">
            <textarea name="message" rows="4" placeholder="Paste an email or message here..."></textarea>
            <br>
            <button type="submit">Check for Spam</button>
        </form>
        
        {% if result %}
        <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 5px;">
            <h3>📨 Message:</h3>
            <p><em>"{{ message }}"</em></p>
            <h3>🔍 Result: <span class="{{ 'spam' if result == 'SPAM' else 'ham' }}">{{ result }}</span></h3>
            <h3>🎯 Confidence: {{ confidence }}</h3>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    message = ""
    confidence = ""
    
    if request.method == 'POST':
        message = request.form['message']
        if message.strip():
            features = vectorizer.transform([message])
            prediction = model.predict(features)[0]
            confidence_score = model.predict_proba(features).max()
            
            result = "SPAM 🚨" if prediction == 1 else "HAM ✅"
            confidence = f"{confidence_score:.1%}"
    
    return render_template_string(HTML_TEMPLATE, result=result, message=message, confidence=confidence)

if __name__ == '__main__':
    print("🌐 Starting web server...")
    print("📱 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0')

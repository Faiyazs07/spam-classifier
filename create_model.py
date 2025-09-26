#!/usr/bin/env python3
"""
Quick script to create and save the spam detection model
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

print("🤖 Creating and saving spam detection model...")

# Create a simple dataset
data = {
    'message': [
        'free prize winner click now', 
        'hello how are you doing',
        'claim your reward urgent', 
        'meeting tomorrow at 3pm',
        'win money fast easy',
        'lunch at noon today',
        'urgent cash prize winner',
        'see you later tonight',
        'congratulations you won',
        'what time is the meeting'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"Dataset created with {len(df)} examples")

# Create and train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

model = MultinomialNB()
model.fit(X, y)

# Save the model
os.makedirs('models', exist_ok=True)

model_data = {
    'model': model,
    'vectorizer': vectorizer
}

joblib.dump(model_data, 'models/best_spam_model.joblib')
print("💾 Model saved to 'models/best_spam_model.joblib'")

# Test the model
test_message = "free prize winner"
features = vectorizer.transform([test_message])
prediction = model.predict(features)[0]
confidence = model.predict_proba(features).max()

print(f"🧪 Test: '{test_message}' -> {'SPAM' if prediction == 1 else 'HAM'} ({confidence:.1%} confidence)")
print("🎉 Model is ready for use!")

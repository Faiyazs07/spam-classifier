#!/usr/bin/env python3
"""
REAL-TIME SPAM PREDICTION TOOL
Usage: python3 predict.py "Your message here"
"""

import joblib
import sys

# Load your trained model
print("🔮 Loading AI Spam Classifier...")
model_data = joblib.load('models/best_spam_model.joblib')
model = model_data['model']
vectorizer = model_data['vectorizer']

def predict_spam(message):
    """Predict if a message is spam or ham"""
    features = vectorizer.transform([message])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features).max()
    
    return "SPAM 🚨" if prediction == 1 else "HAM ✅", f"{confidence:.1%}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Predict from command line argument
        message = sys.argv[1]
        result, confidence = predict_spam(message)
        print(f"\n📨 Message: '{message}'")
        print(f"🔍 Prediction: {result}")
        print(f"🎯 Confidence: {confidence}")
    else:
        # Interactive mode
        print("\n🤖 AI Spam Detector - Interactive Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            message = input("Enter a message to check: ")
            if message.lower() in ['quit', 'exit', 'q']:
                break
            result, confidence = predict_spam(message)
            print(f"🔍 Result: {result} (Confidence: {confidence})")
            print()

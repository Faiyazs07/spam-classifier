"""
Model Training Module
Created by [FAIYAZ SAYED]
My implementation and comparison of spam classification models
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class ModelTrainer:
    """
    My model training and evaluation class
    I tested multiple algorithms to understand their performance differences
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        print("✅ ModelTrainer initialized - Ready to build AI models!")
    
    def train_models(self, X_train, y_train):
        """
        My model training function
        I decided to compare Naive Bayes vs Logistic Regression
        """
        print("🤖 Training multiple models for comparison...")
        
        # Model 1: Naive Bayes - known to work well for text classification
        print("Training Naive Bayes model...")
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        self.models['Naive_Bayes'] = nb_model
        print("📍 Naive Bayes model trained!")
        
        # Model 2: Logistic Regression - good baseline
        print("Training Logistic Regression model...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        self.models['Logistic_Regression'] = lr_model
        print("📍 Logistic Regression model trained!")
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """
        My evaluation function - I wanted to compare performance systematically
        """
        print("\n📊 Evaluating model performance...")
        
        best_score = 0
        best_model_name = None
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy - my primary metric
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            
            # Show classification report for detailed metrics
            print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
            
            # Update best model
            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name
                self.best_model = model
        
        print(f"\n🏆 Best Model: {best_model_name} with {best_score:.4f} accuracy!")
        return best_model_name, self.best_model
    
    def create_confusion_matrix(self, model, X_test, y_test, model_name):
        """
        My visualization function - I wanted to see where the model confuses classes
        """
        # Create models directory if it doesn't exist
        os.makedirs('../models', exist_ok=True)
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], 
                   yticklabels=['Ham', 'Spam'])
        plt.title(f'My {model_name} Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'../models/{model_name}_confusion_matrix.png')
        plt.close()
        
        print(f"📈 Confusion matrix saved for {model_name}!")
    
    def save_best_model(self, vectorizer, filename='best_spam_model.joblib'):
        """
        Save my trained model for future use
        """
        if self.best_model is None:
            print("⚠️ No model trained yet!")
            return
        
        model_data = {
            'model': self.best_model,
            'vectorizer': vectorizer
        }
        
        joblib.dump(model_data, f'../models/{filename}')
        print(f"💾 Best model saved as {filename}!")

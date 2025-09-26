"""
Text Preprocessing Module
Created by [FAIYAZ SAYED]
Custom text cleaning and feature extraction methods I implemented
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class TextPreprocessor:
    """
    My custom text preprocessing class
    I built this to understand how text gets converted to machine-readable features
    """
    
    def __init__(self):
        self.vectorizer = None
        print("✅ TextPreprocessor ready - Let's clean some text!")
    
    def custom_clean_text(self, text):
        """
        My custom text cleaning function
        I experimented with different regex patterns and this worked best
        """
        if isinstance(text, str):
            # Convert to lowercase - standard practice in NLP
            text = text.lower()
            
            # Remove special characters and digits - my choice after testing
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        return ""
    
    def prepare_features(self, df, test_size=0.2, random_state=42):
        """
        My feature preparation pipeline
        This is where I convert text to numbers using CountVectorizer
        """
        print("🧽 Cleaning text messages...")
        df['cleaned_message'] = df['message'].apply(self.custom_clean_text)
        
        # Check if cleaning worked
        print("Sample cleaned messages:")
        for i in range(min(3, len(df))):
            print(f"Original: {df['message'].iloc[i][:50]}...")
            print(f"Cleaned: {df['cleaned_message'].iloc[i][:50]}...")
            print("---")
        
        # Split the data - I'm using stratification to maintain balance
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_message'], 
            df['label_num'], 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['label_num']
        )
        
        print("🔢 Converting text to features...")
        # I chose CountVectorizer for simplicity and effectiveness
        self.vectorizer = CountVectorizer(max_features=3000)
        X_train_features = self.vectorizer.fit_transform(X_train)
        X_test_features = self.vectorizer.transform(X_test)
        
        print("✨ Feature preparation complete!")
        print(f"Training features shape: {X_train_features.shape}")
        print(f"Testing features shape: {X_test_features.shape}")
        
        return X_train_features, X_test_features, y_train, y_test

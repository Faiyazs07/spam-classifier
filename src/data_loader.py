"""
Data Loading Module for Spam Classifier
Created by [FAIYAZ SAYED]
Project: AI Spam Detector - Built from scratch
"""

import pandas as pd
import numpy as np

class DataLoader:
    """
    My custom class to handle dataset loading and initial processing
    I built this to understand how data pipelines work
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        print("✅ DataLoader initialized - Ready to load spam dataset!")
    
    def load_data(self):

    """
    I'm loading the SMS Spam dataset
    This handles the tab-separated format and encoding issues
    """

    try:
        # Try reading as tab-separated (TSV) first - this usually works
        self.df = pd.read_csv(self.file_path, sep='\t', header=None, names=['label', 'message'])
        print("📊 Dataset loaded as TSV successfully!")
        print(f"Shape: {self.df.shape}")
        return self.df
    except Exception as e:
        print(f"❌ TSV loading failed: {e}")
        
        # Try comma-separated with error handling
        try:
            self.df = pd.read_csv(self.file_path, encoding='latin-1', on_bad_lines='skip')
            print("📊 Dataset loaded as CSV with error handling!")
            return self.df
        except Exception as e2:
            print(f"❌ CSV loading failed: {e2}")
            
            # Last attempt: manual reading
            try:
                with open(self.file_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
                
                # Simple manual parsing
                data = []
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        data.append([parts[0], '\t'.join(parts[1:])])
                
                self.df = pd.DataFrame(data, columns=['label', 'message'])
                print("📊 Dataset loaded with manual parsing!")
                return self.df
            except Exception as e3:
                print(f"❌ All loading methods failed: {e3}")
                return None    
    def clean_data(self):
        """
        My data cleaning process 
        I noticed the dataset needed column cleanup and label mapping
        """
        if self.df is None:
            print("⚠️ No data to clean!")
            return None
        
        # Display original columns to understand the structure
        print(f"Original columns: {self.df.columns.tolist()}")
        
        # Keep only first two columns (label and message)
        self.df = self.df.iloc[:, :2]
        
        # Rename columns to something meaningful
        self.df.columns = ['label', 'message']
        
        # Add my custom numeric label column
        self.df['label_num'] = self.df['label'].map({'ham': 0, 'spam': 1})
        
        print("🧹 Data cleaned and prepared!")
        print(f"Spam count: {len(self.df[self.df['label_num'] == 1])}")
        print(f"Ham count: {len(self.df[self.df['label_num'] == 0])}")
        
        return self.df

# Test the class
if __name__ == "__main__":
    print("Testing my DataLoader class...")
    loader = DataLoader('../data/spam.csv')
    data = loader.load_data()
    if data is not None:
        clean_data = loader.clean_data()
        print("DataLoader test completed successfully!")

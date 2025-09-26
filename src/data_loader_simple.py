"""
Simple Data Loader - Guaranteed to work
Created by [FAIYAZ SAYED]
"""

import pandas as pd

class SimpleDataLoader:
    """
    My simple guaranteed-to-work data loader
    """
    
    def __init__(self):
        self.df = None
        print("✅ SimpleDataLoader ready!")
    
    def create_simple_dataset(self):
        """
        I'll create a small but realistic dataset manually
        This ensures we can continue with the project
        """
        data = {
            'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
            'message': [
                'Free prize! Click here to win $1000',
                'Hey, how are you doing today?',
                'URGENT: You have won a luxury car',
                'Lets meet for lunch tomorrow at 1 PM',
                'Congratulations! You won 2 free tickets',
                'Did you finish the homework assignment?',
                'Claim your iPhone now limited time offer',
                'Can you send me the project files?'
            ]
        }
        
        self.df = pd.DataFrame(data)
        self.df['label_num'] = self.df['label'].map({'ham': 0, 'spam': 1})
        
        print("📊 Created simple dataset with 8 examples!")
        print(f"Spam: {len(self.df[self.df['label_num'] == 1])}, Ham: {len(self.df[self.df['label_num'] == 0])}")
        
        return self.df

if __name__ == "__main__":
    loader = SimpleDataLoader()
    data = loader.create_simple_dataset()
    print("\nSample data:")
    print(data.head())

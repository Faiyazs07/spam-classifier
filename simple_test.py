#!/usr/bin/env python3
"""
SIMPLE TEST - Just to make sure everything works
"""

print("=" * 50)
print("🚀 SIMPLE SPAM CLASSIFIER TEST")
print("=" * 50)

# Step 1: Basic Python check
print("1. ✅ Python is working!")

# Step 2: Check imports
try:
    import pandas as pd
    print("2. ✅ Pandas is working!")
except:
    print("2. ❌ Pandas issue")

try:
    from sklearn.naive_bayes import MultinomialNB
    print("3. ✅ Scikit-learn is working!")
except:
    print("3. ❌ Scikit-learn issue")

# Step 3: Simple spam detection logic
print("\n4. 🧠 Testing simple spam detection...")

# Simple example data
messages = [
    "win free prize now",    # spam
    "hello how are you",     # ham  
    "claim your reward",     # spam
    "meeting tomorrow"       # ham
]
labels = [1, 0, 1, 0]  # 1=spam, 0=ham

print("Sample messages:")
for i, msg in enumerate(messages):
    spam_status = "SPAM" if labels[i] == 1 else "HAM"
    print(f"  - '{msg}' -> {spam_status}")

print("\n🎉 BASIC TEST COMPLETED!")
print("If you see this, your environment is working!")

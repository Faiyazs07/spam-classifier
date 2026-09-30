# Spam Classifier

A small Python machine-learning project for classifying short messages as spam or legitimate (“ham”). It includes a command-line demo, model-training scripts, and a lightweight Flask interface.

## What it demonstrates

- Text feature extraction with scikit-learn
- Multinomial Naive Bayes classification
- Saving and loading trained models with Joblib
- A simple browser interface for interactive predictions

This is an educational project built around a compact example dataset. It is useful for exploring the end-to-end workflow, but it is not intended to filter production email.

## Quick start

```bash
git clone https://github.com/Faiyazs07/spam-classifier.git
cd spam-classifier
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and run the command-line example:

```bash
pip install -r requirements.txt
python main.py
```

## Web interface

The Flask app expects a trained model at `models/best_spam_model.joblib`. Generate the model with the repository’s training script, then run:

```bash
python web_app.py
```

Open [http://localhost:5000](http://localhost:5000).

## Project structure

- `main.py` — compact end-to-end example
- `create_model.py` — model-training workflow
- `predict.py` — prediction helper
- `web_app.py` — Flask interface
- `simple_test.py` — basic smoke test
- `src/` — supporting project code

## Notes

Predictions depend heavily on the training data. Before adapting this project for a real application, use a representative dataset, add repeatable evaluation, and review privacy and bias risks.

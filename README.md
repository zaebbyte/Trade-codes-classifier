# TRADE CODES CLASSIFIER - HS Code AI Classifier

##  Overview 🎐
The **HS Code AI Classifier** is a machine learning-based system that predicts **6-digit Harmonized System (HS) codes** from product descriptions.  
It supports **Top-3 predictions with confidence scores**, integrates with API stubs for frontend use, and maintains logs for traceability.

This project was developed as part of my **one-month internship** in Artificial Intelligence and Software Development.

---

## ✨ Features
- Preprocessing of product descriptions (tokenization, lemmatization, stopword removal).
- Baseline ML models: **Naive Bayes** & **Logistic Regression**.
- Embedding-based semantic retrieval using **BERT**.
- **Top-3 HS6 predictions with confidence scores**.
- API stub to simulate frontend integration.
- Logging system for model evaluation and predictions.
- Modular project structure for scalability.

---
## Training Baseline Models
- python ml/train_baseline.py

## Running Predictions
- python api/predict_stub.py


## Example output ⚙️:

{
  "query": "wireless bluetooth earphones",
  "predictions": [
    {"hs6": "851830", "confidence": 0.92, "description": "Headphones and earphones"},
    {"hs6": "851821", "confidence": 0.05, "description": "Loudspeakers"},
    {"hs6": "851890", "confidence": 0.03, "description": "Parts of microphones"}
  ]
}

## Future Improvements 📌
- Finalize XGBoost integration for better accuracy.
- Deploy as a REST API with Flask/FastAPI.
- Build a simple UI for user-friendly predictions.
- Expand dataset with international HS code mappings (WCO, USHTS, ICEGATE).

##👩‍💻 Author

Varsha S T
B.Tech in Computer Science & Engineering | AI Engineer

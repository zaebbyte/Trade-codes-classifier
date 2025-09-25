import joblib
import numpy as np
import os
import pandas as pd

# Load model and vectorizer
MODEL_DIR = "ml/models"
nb_model = joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl"))
tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
print("models loaded ⏳")

# Load reference HS6 → description mapping
DATA_PATH = "data/hs_tree_cleaned.csv"
hs_data = pd.read_csv(DATA_PATH)
hs6_to_desc = hs_data.drop_duplicates("HS6").set_index("HS6")["description"].to_dict()
print("description mapped📌")   

def predict_top_k(description, model=nb_model, vectorizer=tfidf_vectorizer, k=3):
    """
    Predict top-k HS6 codes with confidence scores and descriptions.
    
    Returns:
        List of dicts: [{hs6, confidence, description}, ...]
    """
    if not description.strip():
        return []

    # Vectorize the input
    X_input = vectorizer.transform([description])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]
        top_k_indices = np.argsort(proba)[::-1][:k]
        top_k_labels = model.classes_[top_k_indices]
        top_k_scores = proba[top_k_indices]

        results = []
        for hs6, score in zip(top_k_labels, top_k_scores):
            desc = hs6_to_desc.get(hs6, "N/A")
            results.append({"hs6": hs6, "confidence": round(score, 4), "description": desc})
        return results
    else:
        label = model.predict(X_input)[0]
        desc = hs6_to_desc.get(label, "N/A")
        return [{"hs6": label, "confidence": 1.0, "description": desc}]

# Example test
if __name__ == "__main__":
    sample = "toys for kids"
    results = predict_top_k(sample)
    print("🔍 Top 3 HS6 Predictions:")
    for r in results:
        print(f"  → {r['hs6']} ({r['confidence']*100:.2f}%) - {r['description']}")

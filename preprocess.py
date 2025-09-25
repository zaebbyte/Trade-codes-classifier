import os
import pandas as pd
import spacy
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA_PATH = "data/HS_Tree.csv"
OUTPUT_PATH = "data/hs_tree_cleaned.csv"
LOG_PATH = "logs/model_baseline_results.txt"
MODEL_DIR = "ml/models"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("🔄 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["description", "HS6"])

print("🧹 Preprocessing text with spaCy...")
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(str(text).lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

df["clean_description"] = df["description"].apply(clean_text)

#Save cleaned data
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Cleaned data saved to {OUTPUT_PATH}")

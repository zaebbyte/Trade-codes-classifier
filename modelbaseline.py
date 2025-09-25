import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_and_save_baselines(csv_path, model_dir, log_path):
    df = pd.read_csv("data/hs_tree_cleaned.csv")
    df = df.dropna(subset=["clean_description"])
    df = df[df["clean_description"].str.strip() != ""]

    X = df["clean_description"]
    y = df["HS6"]

    print("Nulls:", X.isnull().sum())
    print("Empty rows:", (X.str.strip() == "").sum())


    tfidf = TfidfVectorizer(max_features=1000)
    X_vec = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    nb = MultinomialNB().fit(X_train, y_train)
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    y_pred_nb = nb.predict(X_test)
    y_pred_lr = lr.predict(X_test)

    report_nb = classification_report(y_test, y_pred_nb)
    report_lr = classification_report(y_test, y_pred_lr)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(nb, os.path.join(model_dir, "naive_bayes.pkl"))
    joblib.dump(lr, os.path.join(model_dir, "logistic_regression.pkl"))
    joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

    with open(log_path, "w") as log_file:
        log_file.write("=== NAIVE BAYES REPORT ===\n" + report_nb)
        log_file.write("\n\n=== LOGISTIC REGRESSION REPORT ===\n" + report_lr)

    print("✅ Models and logs saved.")

if __name__ == "__main__":
    # Define paths
    csv_path = "data/hs_tree_cleaned.csv"
    model_dir = "ml/models"
    log_path = "logs/model_baseline_results.txt"

    # Run the training pipeline
    train_and_save_baselines(csv_path, model_dir, log_path)

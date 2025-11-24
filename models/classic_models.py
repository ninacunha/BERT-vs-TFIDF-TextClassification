import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import seaborn as sns

def run_classic_models(csv_path="spam.csv"):
    df = pd.read_csv(csv_path, encoding='latin-1')

    # Drop unused columns
    df = df[['v1','v2']]
    df.rename(columns={'v1':'target','v2':'text'}, inplace=True)

    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])

    # Remove duplicates
    df = df.drop_duplicates()

    def clean_text(text):
        text = text.lower()
        return "".join([ch if ch.isalnum() else " " for ch in text])

    df['transformed_text'] = df['text'].apply(clean_text)

    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['transformed_text']).toarray()
    y = df['target'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    clfs = {
        'Logistic Regression': LogisticRegression(solver='liblinear', penalty='l1'),
        'Decision Tree': DecisionTreeClassifier(max_depth=5),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in clfs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)

        print(f"\n===== {name} =====")
        print("Accuracy:", acc)
        print("Precision:", prec)

        # Heatmap
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        results[name] = {"accuracy": acc, "precision": prec}

    return results

if __name__ == "__main__":
    run_classic_models()

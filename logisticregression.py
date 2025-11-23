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
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_csv('spam.csv', encoding='latin-1')

#drop columns
df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace=True)

encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

df.isnull().sum()

df.duplicated().sum()

df = df.drop_duplicates(keep= 'first')

def transform_text(text):
    text = text.lower()
    cleaned = "".join([ch if ch.isalnum() else " " for ch in text])
    return cleaned

df['transformed_text'] = df['text'].apply(transform_text)

tfid = TfidfVectorizer(max_features=3000)

#dependent and independent variables that we then split into train and test data
X = tfid.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

#training and testing the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

logisticregression = LogisticRegression(solver='liblinear', penalty='l1')
decisiontree = DecisionTreeClassifier(max_depth=5)
gradientboosting = GradientBoostingClassifier(n_estimators=100, random_state=42)
xgboost = XGBClassifier(n_estimators = 100, random_state=42)

clfs = {
    'LR': logisticregression,
    'DT': decisiontree,
    'XGB': xgboost,
    'GB': gradientboosting
}

from sklearn.metrics import accuracy_score, precision_score
def classification(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision, y_pred

accuracies = []
precisions = []

for name, model in clfs.items():
    current_accuracy, current_precision, y_pred = classification(model, X_train, y_train, X_test, y_test)
    print()
    print("For this model: ", name)
    print("Accuracy is: ", current_accuracy)
    print("Precision is: ", current_precision)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    accuracies.append(current_accuracy)
    precisions.append(current_precision)

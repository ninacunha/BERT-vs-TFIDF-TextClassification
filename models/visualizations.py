import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from wordcloud import WordCloud
import pandas as pd

def plot_roc(y_test, y_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

def generate_wordcloud(csv_path="spam.csv"):
    df = pd.read_csv(csv_path, encoding='latin-1')[["v1","v2"]]
    df.columns = ["label","text"]

    spam_messages = " ".join(df[df["label"]=="spam"]["text"])

    wc = WordCloud(width=800, height=400, background_color="white").generate(spam_messages)

    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Common Spam Words")
    plt.show()

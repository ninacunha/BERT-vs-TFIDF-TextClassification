from models.naive_bayes import train_naive_bayes
from models.bert_classifier import train_bert
from models.classic_models import run_classic_models

print("\n=== RUNNING CLASSIC ML MODELS ===")
run_classic_models()

print("\n=== RUNNING NAIVE BAYES ===")
train_naive_bayes()

print("\n=== RUNNING BERT CLASSIFIER ===")
train_bert()

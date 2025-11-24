from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    print("\n===== BERT RESULTS =====")
    print("\nConfusion Matrix:\n", confusion_matrix(labels, preds))
    print("\nClassification Report:\n", classification_report(labels, preds))

    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

def train_bert():
    print("Loading SMS Spam dataset...")

    # Using HuggingFace Dataset version of SMS Spam dataset
    dataset = load_dataset("sms_spam")

    # Train/test split
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["sms"], padding=True, truncation=True, max_length=128)

    print("Tokenizing data...")
    dataset = dataset.map(tokenize, batched=True)

    # Prepare labels + tensors
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./bert_output",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )

    print("\nTraining BERT...")
    trainer.train()

    print("\nEvaluating...")
    trainer.evaluate()

    print("\nBERT training complete.")
    return model, tokenizer

if __name__ == "__main__":
    train_bert()

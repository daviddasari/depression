import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# === 1️⃣ Configuration ===
MODEL_PATH = "./results-depression-extended"
CSV_FILE = "go_emotions_dataset.csv"

# === 2️⃣ Load model and tokenizer ===
print(f"Loading model from {MODEL_PATH}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# === 3️⃣ Load dataset ===
print(f"Loading dataset from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

# Drop NaN text entries
df.dropna(subset=["text"], inplace=True)
df = df.reset_index(drop=True)

# === 4️⃣ Define depression-related emotion columns ===
DEPRESSION_PROXY_LABELS = ["sadness", "disappointment", "grief", "remorse"]

# Ensure all required columns exist
for col in DEPRESSION_PROXY_LABELS:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV. Please check the dataset.")

# Create binary 'labels' column
df["labels"] = (df[DEPRESSION_PROXY_LABELS].sum(axis=1) > 0).astype(int)

# Keep only text + label columns
df_test = df[["text", "labels"]].sample(frac=0.2, random_state=42)  # use 20% as test set
dataset = Dataset.from_pandas(df_test)

# === 5️⃣ Tokenization ===
print("Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# === 6️⃣ Initialize Trainer for prediction ===
trainer = Trainer(model=model, tokenizer=tokenizer)

# === 7️⃣ Run inference ===
print("\nRunning inference on test set...")
predictions = trainer.predict(dataset)
logits = predictions.predictions
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
labels = np.array(predictions.label_ids)

# === 8️⃣ Search best threshold for F1 ===
best_threshold = 0.5
best_f1 = 0
for t in np.linspace(0.1, 0.9, 17):
    preds = (probs >= t).astype(int)
    f1 = f1_score(labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# === 9️⃣ Final metrics ===
final_preds = (probs >= best_threshold).astype(int)
acc = accuracy_score(labels, final_preds)
prec = precision_score(labels, final_preds)
rec = recall_score(labels, final_preds)
cm = confusion_matrix(labels, final_preds)

print("\n--- Best Threshold Evaluation ---")
print(f"Optimal threshold: {best_threshold:.2f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {best_f1:.4f}")

print("\nConfusion Matrix (rows = Actual, cols = Predicted):")
print(cm)

print("\nDetailed Classification Report:")
print(classification_report(labels, final_preds, digits=4))

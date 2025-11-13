# -----------------------------------------------------------
# Depression Proxy Detector — DistilBERT (complete script)
# - Stratified split
# - Tokenization (max_length=256)
# - Modern Trainer args (eval_strategy/save_strategy)
# - Early stopping (patience=2)
# - Best checkpoint by F1
# -----------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    pipeline,
    set_seed,
)

# -----------------------------
# 1) Config
# -----------------------------
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./results-depression-extended"

# Emotions we’ll treat as “depression proxy” (positive class = 1)
DEPRESSION_PROXY_LABELS = [
    "sadness", "grief", "disappointment", "remorse",
    "anger", "annoyance", "fear", "nervousness", "disgust"
]

id2label = {0: "NOT_DEPRESSED_PROXY", 1: "DEPRESSED_PROXY"}
label2id = {"NOT_DEPRESSED_PROXY": 0, "DEPRESSED_PROXY": 1}

set_seed(42)


# -----------------------------
# 2) Data loading & prep
# -----------------------------
def load_and_prepare_data_for_hf(csv_path: str = "go_emotions_dataset.csv"):
    """Load CSV, build binary label, return HF DatasetDict with stratified split."""
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found.")
        return None

    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        print("Error: The CSV file is missing a 'text' column.")
        return None

    print("Formatting data for Depression Proxy (Binary) classification...")
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    # Positive label if any depression-proxy emotion is present on the row
    missing = [c for c in DEPRESSION_PROXY_LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected emotion columns: {missing}")

    df["labels"] = (df[DEPRESSION_PROXY_LABELS].sum(axis=1) > 0).astype(int)

    print("\n--- New Label Distribution ---")
    print(df["labels"].value_counts(normalize=True).rename("proportion"))
    print("--------------------------------\n")

    df_final = df[["text", "labels"]]

    # Stratified split to preserve class ratio in train/test
    train_df, test_df = train_test_split(
        df_final, test_size=0.2, stratify=df_final["labels"], random_state=42
    )

    datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "test":  Dataset.from_pandas(test_df.reset_index(drop=True)),
    })

    print("Data preparation complete.")
    return datasets


# -----------------------------
# 3) Tokenization
# -----------------------------
def get_tokenized_datasets(all_datasets, tokenizer):
    print("Tokenizing data...")

    def preprocess(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=True,
            max_length=256,  # a bit longer to catch subtle cues later in text
        )

    tokenized = all_datasets.map(preprocess, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")

    print("Tokenization complete.")
    return tokenized


# -----------------------------
# 4) Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}


# -----------------------------
# 5) Main training
# -----------------------------
def main():
    # Load/prepare data
    datasets = load_and_prepare_data_for_hf()
    if datasets is None:
        return

    # Tokenizer + tokenized datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = get_tokenized_datasets(datasets, tokenizer)

    # Base model
    print(f"Loading pre-trained model: '{MODEL_NAME}'...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    # Collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer)

    # ---- Training args (modern) ----
    # NOTE: Use eval_strategy (new name). save_strategy etc. remain the same.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,                # train up to 8; early stopping will cut sooner
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",             # <--- new argument name
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=200,
        fp16=torch.cuda.is_available(),    # mixed precision on GPU
        report_to="none",                  # disable external loggers (W&B/MLflow)
    )

    # Trainer + early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print("\nStarting extended training (early stopping enabled)...\n")
    trainer.train()
    print("\nTraining complete!")

    # Evaluate
    print("\nEvaluating best model on test set...")
    eval_results = trainer.evaluate()
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {eval_results.get('eval_accuracy', float('nan')):.4f}")
    print(f"F1-Score: {eval_results.get('eval_f1', float('nan')):.4f}")

    # Save
    print("\nSaving best model to:", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)

    # Quick sanity test
    print("\n--- Model Test (Sample Sentences) ---")
    test_texts = [
        "I feel so empty and sad all the time.",
        "Nothing feels good anymore. I just want to sleep.",
        "I'm just so angry at everyone for no reason.",
        "This is amazing, I'm so happy!",
        "That was a good movie, I really enjoyed it.",
        "I'm not sure what to think about that.",
    ]
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
    for t in test_texts:
        p = clf(t)[0]
        print(f"Text: {t!r}")
        print(f"  -> Prediction: {p['label']} (Score: {p['score']:.4f})\n")


# -----------------------------
# 6) Run
# -----------------------------
if __name__ == "__main__":
    main()

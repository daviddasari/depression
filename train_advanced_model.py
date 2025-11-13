import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, hamming_loss
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. Setup & Configuration ---
MODEL_NAME = 'distilbert-base-uncased'

# Define the 28 emotion labels, same as before.
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]


# --- 2. Data Loading & Preparation for Hugging Face ---
def load_and_prepare_data_for_hf():
    """Loads the CSV and formats it for Hugging Face multi-label training."""
    file_name = 'go_emotions_dataset.csv'

    try:
        print(f"Loading data from '{file_name}'...")
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: '{file_name}' not found.")
        print("Please make sure 'go_emotions_dataset.csv' is in the same directory.")
        return None
    except Exception as e:
        print(f"An error occurred loading the file: {e}")
        return None

    # Check for 'text' column
    if 'text' not in df.columns:
        print("Error: The CSV file is missing a 'text' column.")
        return None

    print("Formatting data for Hugging Face...")
    # Drop rows where 'text' is missing
    df.dropna(subset=['text'], inplace=True)
    df = df.reset_index(drop=True)

    # Create the 'labels' column
    labels = df[EMOTION_LABELS].values.astype(float).tolist()
    df['labels'] = labels

    # Keep only the 'text' and 'labels' columns
    df_final = df[['text', 'labels']]

    # Convert the Pandas DataFrame to a Hugging Face Dataset
    hg_dataset = Dataset.from_pandas(df_final)

    # Create our own train/test split (80% train, 20% test)
    print("Splitting data into train/test sets...")
    all_datasets = hg_dataset.train_test_split(test_size=0.2, seed=42)

    print("Data preparation complete.")
    return all_datasets  # This is now a DatasetDict


# --- 3. Tokenization ---
def get_tokenized_datasets(all_datasets, tokenizer):
    """Tokenizes the text data."""
    print("Tokenizing data...")

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

    tokenized_datasets = all_datasets.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    tokenized_datasets.set_format('torch')

    print("Tokenization complete.")
    return tokenized_datasets


# --- 4. Metrics Calculation ---
def compute_metrics(eval_pred):
    """Calculates metrics for multi-label classification."""
    logits, labels = eval_pred
    sigmoid = 1.0 / (1.0 + np.exp(-logits))
    predictions = (sigmoid > 0.5).astype(int)
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    hamming_avg = hamming_loss(labels, predictions)

    return {
        'f1_micro': f1_micro,
        'hamming_loss': hamming_avg
    }


# --- 5. Main Training Function ---
def main():
    # Load and prepare data
    all_datasets = load_and_prepare_data_for_hf()
    if all_datasets is None:
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize datasets
    tokenized_datasets = get_tokenized_datasets(all_datasets, tokenizer)

    # Load model
    print(f"Loading pre-trained model: '{MODEL_NAME}'...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(EMOTION_LABELS),
        problem_type="multi_label_classification"
    )

    # --- Training Arguments ---
    # THIS VERSION HAS NO EVALUATION ARGUMENTS TO AVOID THE CRASH
    training_args = TrainingArguments(
        output_dir='./results',  # Where to save the model
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=1000,

        # NOTE: 'evaluate_during_training' and 'evaluation_strategy'
        # are BOTHREMOVED to prevent the TypeError.
        # We will evaluate manually at the end.

        load_best_model_at_end=False,  # Must be False if no evaluation
    )

    # --- Initialize the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # --- Train the Model ---
    print("\nStarting model training...")
    print("This will take a long time without a GPU. (e.g., 10-20 min on a T4 GPU, 2-3+ hours on a CPU)")

    trainer.train()

    print("\nTraining complete!")

    # --- Evaluate the Model ---
    print("\nEvaluating final model on the test set...")
    eval_results = trainer.evaluate()

    print("\n--- Evaluation Results ---")
    print(f"F1-Score (Micro): {eval_results['eval_f1_micro']:.4f}")
    print(f"Hamming Loss: {eval_results['eval_hamming_loss']:.4f} (Lower is better)")

    # --- 9. Test with custom text ---
    print("\n--- Model Test (Showing Top 3 Predictions) ---")
    test_texts = [
        "This is amazing, I'm so happy!",
        "I'm really angry and disappointed about this.",
        "I'm not sure what to think about that.",
        "That's so sad, I'm sending my love."
    ]

    print("Creating prediction pipeline...")
    from transformers import pipeline

    model.to('cpu')
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    for text in test_texts:
        print(f"\nText: '{text}'")
        raw_preds = pipe(text)[0]

        # Filter for our 28 emotion labels
        emotion_preds = [
            pred for pred in raw_preds.copy()
            if pred['label'] in EMOTION_LABELS
        ]

        # Sort by score, descending
        emotion_preds.sort(key=lambda x: x['score'], reverse=True)

        # Get the Top 3
        top_3_preds = emotion_preds[:3]

        print("Top 3 Predictions:")
        for pred in top_3_preds:
            print(f"  - {pred['label']}: {pred['score']:.4f}")


if __name__ == "__main__":
    main()
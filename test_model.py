import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. Configuration ---

# This MUST be the folder where your model saved.
MODEL_DIR = "./results/checkpoint-10562"

# Define the 28 emotion labels
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# --- NEW: Create label mapping dictionaries ---
# This tells the model what string name corresponds to which output number
id2label = {i: label for i, label in enumerate(EMOTION_LABELS)}
label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
# --- END NEW ---

# The sentences we want to test
TEST_TEXTS = [
    "This is amazing, I'm so happy!",
    "I'm really angry and disappointed about this.",
    "I'm not sure what to think about that.",
    "That's so sad, I'm sending my love."
]


# --- 2. Main Test Function ---
def main():
    print(f"Loading saved model from: '{MODEL_DIR}'...")

    # Check if the directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory not found at '{MODEL_DIR}'")
        print("Please check the 'results' folder for a 'checkpoint-XXXXX' subfolder.")
        return

    try:
        # Load the tokenizer and model from your saved results
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        # --- MODIFIED: Pass the label dictionaries to the model ---
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            id2label=id2label,
            label2id=label2id
        )
        # --- END MODIFIED ---

    except Exception as e:
        print(f"Error loading model from '{MODEL_DIR}'.")
        print("It's possible the saved model is corrupted or from a different transformers version.")
        print(f"Error details: {e}")
        return

    print("Model loaded successfully.")
    print("Creating prediction pipeline...")

    # Create a pipeline to make predictions easy
    # We remove .to('cpu') so the pipeline can use your GPU
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True  # Get scores for all labels
    )

    # --- 3. Run Model Test ---
    print("\n--- Model Test (Showing Top 3 Predictions) ---")

    for text in TEST_TEXTS:
        print(f"\nText: '{text}'")

        # Get the raw predictions from the pipeline
        raw_preds = pipe(text)[0]

        # --- THIS FILTERING WILL NOW WORK ---
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
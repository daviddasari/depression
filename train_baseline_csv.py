import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, hamming_loss, f1_score

# --- 1. Setup & Configuration ---

# Define the 28 emotion labels. These MUST match the column names in your CSV.
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]


# --- 2. NLTK Downloader ---
def setup_nltk():
    """Downloads necessary NLTK models."""
    try:
        nltk.data.find('corpus/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')  # Check for the new resource
        print("NLTK resources (stopwords, punkt, punkt_tab) are already downloaded.")
    except LookupError:
        print("Downloading NLTK data (stopwords, punkt, punkt_tab)...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # Add this line
        print("NLTK data downloaded.")


# Get English stopwords
stop_words = set(stopwords.words('english'))


# --- 3. Text Preprocessing ---
def clean_text(text):
    """Cleans and preprocesses a single text string."""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    return " ".join(cleaned_tokens)


# --- 4. Data Loading & Preparation ---
def load_and_prepare_data():
    """Loads and prepares data from the single 'go_emotions_dataset.csv' file."""

    file_name = 'go_emotions_dataset.csv'

    try:
        print(f"Loading data from '{file_name}'...")
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: '{file_name}' not found.")
        print("Please make sure your file is in the same directory as this script")
        print("and is named exactly 'go_emotions_dataset.csv'.")
        return None
    except Exception as e:
        print(f"An error occurred loading the file: {e}")
        return None

    print(f"Total rows loaded: {len(df)}")

    # Check if 'text' column exists
    if 'text' not in df.columns:
        print("Error: The CSV file is missing a 'text' column.")
        return None

    # Drop rows where 'text' is missing (if any)
    df.dropna(subset=['text'], inplace=True)

    print("Cleaning text data...")
    # Apply the text cleaning function
    df['text_cleaned'] = df['text'].apply(clean_text)

    print("Formatting data for model...")
    # X (features) is the cleaned text
    X = df['text_cleaned']

    # y (labels) is the matrix of 0s and 1s from the emotion columns
    try:
        y = df[EMOTION_LABELS]
    except KeyError as e:
        print(f"Error: Missing an emotion column: {e}")
        print("Please check that all emotion labels in the EMOTION_LABELS list")
        print("match the column names in your CSV file *exactly*.")
        return None

    print("Splitting data into training and test sets (80/20 split)...")
    # Create our own train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Data preparation complete.")
    return X_train, X_test, y_train, y_test


# --- 5. Main Model Training & Evaluation ---
def main():
    """Main function to run the full pipeline."""
    setup_nltk()

    # Load and prepare data
    data = load_and_prepare_data()

    if data is None:
        return  # Stop if data loading failed

    X_train, X_test, y_train, y_test = data

    # --- 6. Create the Model Pipeline ---
    print("\nBuilding model pipeline...")

    # 1. Text Vectorizer: Converts text into a TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    # 2. Classifier: We use Logistic Regression as the base
    # We wrap it in MultiOutputClassifier to handle multi-label predictions
    base_classifier = LogisticRegression(solver='liblinear', C=1.0, random_state=42)
    multi_label_classifier = MultiOutputClassifier(base_classifier, n_jobs=-1)

    # Create the full pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('clf', multi_label_classifier)
    ])

    # --- 7. Train the Model ---
    print("Training the model... (This may take a few minutes on all the data)")
    pipeline.fit(X_train, y_train)
    print("Model training complete!")

    # --- 8. Evaluate the Model ---
    print("\nEvaluating model on the test set...")
    y_pred = pipeline.predict(X_test)

    # Print key metrics
    # Note: classification_report can be VERY long. We'll print summary metrics first.
    print("\n--- Key Metrics ---")

    # Hamming Loss: The fraction of labels that are incorrectly predicted. Lower is better.
    hamming_avg = hamming_loss(y_test, y_pred)
    print(f"Hamming Loss: {hamming_avg:.4f} (Lower is better)")

    # F1-Score (Micro): Calculates metrics globally by counting total TPs, FNs, FPs. Good for imbalanced classes.
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    print(f"F1-Score (Micro): {f1_micro:.4f}")

    # F1-Score (Macro): Calculates metrics for each label, and finds their unweighted mean.
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"FF1-Score (Macro): {f1_macro:.4f}")

    # Print a detailed classification report
    print("\n--- Detailed Classification Report (Per-Emotion) ---")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS, zero_division=0))

    # --- 9. Test with custom text ---
    print("\n--- Model Test ---")
    test_texts = [
        "This is amazing, I'm so happy!",
        "I'm really angry and disappointed about this.",
        "I'm not sure what to think about that.",
        "That's so sad, I'm sending my love."
    ]

    # Preprocess the custom texts
    cleaned_test_texts = [clean_text(text) for text in test_texts]

    # Get predictions
    custom_preds = pipeline.predict(cleaned_test_texts)

    for text, labels in zip(test_texts, custom_preds):
        print(f"\nText: '{text}'")

        # 'labels' is now a numpy array of 0s and 1s
        # We find which emotions are '1'
        predicted_emotions = [EMOTION_LABELS[i] for i, label_val in enumerate(labels) if label_val == 1]

        print(f"Predicted Emotions: {', '.join(predicted_emotions) if predicted_emotions else 'None'}")


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import os
import warnings
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io  # --- NEW: For file uploader
import re  # --- NEW: For word cloud text cleaning

# --- NEW: Imports for LIME and WordCloud ---
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud, STOPWORDS

# --- 1️⃣ Configuration ---

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Depression Proxy Detector",
    layout="wide"
)


# --- MODIFIED: Added CSS for LIME ---
def set_custom_style():
    st.markdown("""
        <style>
        /* Import Google Font 'Source Sans Pro' */
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');

        html, body, [class*="st-"] {
           font-family: 'Source Sans Pro', sans-serif;
        }

        h1 {
            font-weight: 700;
            color: #2C3E50;
        }
        h2 {
            font-weight: 600;
            color: #34495E;
        }
        h3 {
            font-weight: 600;
            color: #7F8C8D;
        }

        .st-emotion-cache-1h9usn1 {
            font-size: 1.05rem;
            font-weight: 600;
        }

        /* --- NEW: Style LIME explanations for dark theme --- */
        .lime-text-container {
            color: white; /* Make text readable in dark mode */
        }
        .lime-text-container span {
            opacity: 1.0; /* Ensure all words are visible */
        }
        </style>
    """, unsafe_allow_html=True)


set_custom_style()

MODEL_DIR = "mist01/depression"  # <-- This is now pointing to Hugging Face
DATA_FILE = "go_emotions_dataset.csv"

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

DEPRESSION_PROXY_LABELS = [
    "sadness", "grief", "disappointment", "remorse",
    "anger", "annoyance", "fear", "nervousness", "disgust"
]

id2label = {0: "NOT_DEPRESSED_PROXY", 1: "DEPRESSED_PROXY"}
label2id = {"NOT_DEPRESSED_PROXY": 0, "DEPRESSED_PROXY": 1}
class_names = ["NOT_DEPRESSED_PROXY", "DEPRESSED_PROXY"]


# --- 2️⃣ Utility Functions ---

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}


@st.cache_resource
def load_model_and_pipeline():
    """Load the trained model and tokenizer once from Hugging Face."""
    print(f"Cache miss: Loading model from {MODEL_DIR}...")
    
    try:
        # This will download the model from your Hugging Face repo
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, id2label=id2label, label2id=label2id
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    except Exception as e:
        # If it fails, show an error and return 3 None values
        st.error(f"Fatal Error: Could not load model from '{MODEL_DIR}'. Error: {e}")
        return None, None, None # <-- Returns 3 values on failure

    # Create the pipeline
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1, # <-- Force CPU (-1) for Streamlit Cloud
        return_all_scores=True 
    )

    print("✅ Model and pipeline loaded successfully.")
    return model, tokenizer, pipe


# --- NEW: LIME Explainer Function ---
@st.cache_resource
def get_lime_explainer():
    """Create and cache the LIME text explainer object."""
    print("Cache miss: Creating LIME explainer...")
    return LimeTextExplainer(class_names=class_names)


# --- NEW: LIME Prediction Wrapper ---
def lime_predict_proba(texts, _pipeline):
    """
    Wrapper function to get prediction probabilities in the
    format LIME expects: [prob_class_0, prob_class_1]
    """
    results = _pipeline(texts)
    probs = []
    for res in results:
        # Sort results by label to ensure correct order
        sorted_res = sorted(res, key=lambda x: x['label'])
        # Extract scores in [NOT_DEPRESSED, DEPRESSED] order
        prob_not_depressed = sorted_res[0]['score']
        prob_depressed = sorted_res[1]['score']
        probs.append([prob_not_depressed, prob_depressed])
    return np.array(probs)


@st.cache_data
def load_and_prep_test_data(_tokenizer):
    """Load and preprocess the GoEmotions test data."""
    print("Cache miss: Loading and processing test data...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error(f"File not found: {DATA_FILE}")
        return None

    df.dropna(subset=["text"], inplace=True)
    df["labels"] = (df[DEPRESSION_PROXY_LABELS].sum(axis=1) > 0).astype(int)

    hg_dataset = Dataset.from_pandas(df[["text", "labels"]])
    all_datasets = hg_dataset.train_test_split(test_size=0.2, seed=42)

    def preprocess_function(examples):
        return _tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

    tokenized_test = all_datasets["test"].map(preprocess_function, batched=True)
    tokenized_test = tokenized_test.remove_columns(["text"])
    tokenized_test.set_format("torch")

    print("✅ Test data loaded.")
    return tokenized_test


@st.cache_data
def run_evaluation(_model, _tokenized_test_set):
    """Evaluate model performance once per session."""
    print("Cache miss: Running evaluation...")
    trainer = Trainer(model=_model, compute_metrics=compute_metrics)
    return trainer.evaluate(eval_dataset=_tokenized_test_set)


@st.cache_data
def get_predictions(_model, _tokenized_test_set):
    """Get model predictions for the test set (for confusion matrix)."""
    print("Cache miss: Getting predictions for confusion matrix...")
    trainer = Trainer(model=_model)
    raw_predictions = trainer.predict(_tokenized_test_set)
    return raw_predictions


@st.cache_data
def load_visualization_data():
    """Load and prepare data for visualization."""
    print("Cache miss: Loading visualization data...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error(f"File not found: {DATA_FILE}")
        return None, None, None

    emotion_counts = df[EMOTION_LABELS].sum().sort_values(ascending=False)
    emotion_counts_df = emotion_counts.reset_index(name="count").rename(columns={"index": "Emotion"})

    df["num_labels"] = df[EMOTION_LABELS].sum(axis=1)
    label_counts_df = df["num_labels"].value_counts().sort_index().reset_index(name="count")
    label_counts_df.rename(columns={"num_labels": "Number of Labels"}, inplace=True)

    df["proxy_label_str"] = (df[DEPRESSION_PROXY_LABELS].sum(axis=1) > 0).astype(int)
    df["proxy_label_str"] = df["proxy_label_str"].map({0: "NOT_DEPRESSED_PROXY", 1: "DEPRESSED_PROXY"})
    proxy_counts_df = df["proxy_label_str"].value_counts().reset_index(name="count")

    return emotion_counts_df, label_counts_df, proxy_counts_df


# --- NEW: Function to load data for Word Clouds ---
@st.cache_data
def load_text_data_for_wordcloud():
    """Load text data and split by proxy label for word cloud."""
    print("Cache miss: Loading data for word clouds...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error(f"Could not load data for word clouds: {DATA_FILE}")
        return None, None

    df.dropna(subset=["text"], inplace=True)
    df["labels"] = (df[DEPRESSION_PROXY_LABELS].sum(axis=1) > 0).astype(int)

    # Clean and join text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        return text

    df["cleaned_text"] = df["text"].apply(clean_text)

    proxy_text = " ".join(df[df["labels"] == 1]["cleaned_text"])
    not_proxy_text = " ".join(df[df["labels"] == 0]["cleaned_text"])

    return proxy_text, not_proxy_text


# --- NEW: Function to generate Word Cloud ---
def generate_wordcloud(text_data):
    """Generate a word cloud image from a block of text."""
    if not text_data:
        return None

    stopwords = set(STOPWORDS)
    wc = WordCloud(width=800, height=400, background_color='black',
                   stopwords=stopwords, max_words=100,
                   colormap='viridis').generate(text_data)
    return wc.to_image()


@st.cache_data
def load_raw_data_samples():
    """Load raw data for display on the About tab."""
    print("Cache miss: Loading raw data samples...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        return None, None

    df["labels"] = (df[DEPRESSION_PROXY_LABELS].sum(axis=1) > 0).astype(int)
    proxy_df = df[df["labels"] == 1][["text"]].head()
    not_proxy_df = df[df["labels"] == 0][["text"]].head()
    return proxy_df, not_proxy_df


# --- NEW: Function to process batch file ---
def process_batch_file(file_content, _pipeline):
    """Read file content, run predictions, and return a DataFrame."""
    lines = file_content.decode('utf-8').splitlines()

    # Remove empty lines
    texts = [line.strip() for line in lines if line.strip()]

    if not texts:
        return pd.DataFrame(columns=["text", "prediction", "confidence"])

    # Run pipeline on all texts at once
    predictions = _pipeline(texts)

    # Process results
    results = []
    for text, preds in zip(texts, predictions):
        # Find the prediction with the highest score
        best_pred = max(preds, key=lambda x: x['score'])
        results.append({
            "text": text,
            "prediction": best_pred['label'],
            "confidence": f"{best_pred['score'] * 100:.2f}%"
        })

    return pd.DataFrame(results)


# --- 3️⃣ Sidebar ---
st.sidebar.title("About This Analyzer")
st.sidebar.info(
    """
    **Project:** AI Depression Proxy Detection  
    This model is a **DistilBERT** model fine-tuned on the
    **GoEmotions** dataset to detect depressive emotion proxies.
    """
)
st.sidebar.title("How to Use")
st.sidebar.markdown(
    """
    1. Enter text to analyze in the text box.  
    2. Click the **Analyze** button.  
    3. The model will predict the result.  
    """
)

# --- 4️⃣ Main Tabs ---
st.title("🧠 AI Depression Proxy Detector")
# --- MODIFIED: Added 5th tab ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Analyzer",
    "Visual Insights",
    "Final Evaluation",
    "About This Model",
    "Mental Health Resources"  # --- NEW TAB
])

with st.spinner("Loading model and utilities..."):
    model, tokenizer, classifier_pipeline = load_model_and_pipeline()
    # --- NEW: Load LIME Explainer ---
    if classifier_pipeline:
        lime_explainer = get_lime_explainer()
    else:
        lime_explainer = None

# --- Tab 1: Analyzer (MODIFIED) ---
with tab1:
    if classifier_pipeline is None:
        st.error("Model failed to load. Please verify the model path or check the logs.")
    else:
        st.header("Text Analyzer")

        # --- NEW: Sub-section for single text ---
        st.subheader("Analyze a Single Text")
        text = st.text_area("Enter text to analyze:", "", height=150)

        if st.button("Analyze"):
            if text.strip():
                with st.spinner("Analyzing..."):
                    # Get prediction
                    pred_scores = classifier_pipeline(text)[0]
                    best_pred = max(pred_scores, key=lambda x: x['score'])
                    label, score = best_pred["label"], best_pred["score"]

                    # --- NEW: LIME Explanation ---
                    st.markdown("---")
                    st.subheader("Prediction Explanation (LIME)")


                    # Define the prediction wrapper with the loaded pipeline
                    def predict_proba_wrapper(texts):
                        return lime_predict_proba(texts, classifier_pipeline)


                    # Generate explanation
                    explanation = lime_explainer.explain_instance(
                        text,
                        predict_proba_wrapper,
                        num_features=10,
                        labels=[0, 1]  # Explain both classes
                    )

                    # Get the ID for the predicted class
                    predicted_class_id = label2id[label]

                    # Display the explanation as HTML
                    html = explanation.as_html(labels=[predicted_class_id])
                    # Inject custom class for styling
                    html = f"<div class='lime-text-container'>{html}</div>"
                    st.components.v1.html(html, height=250, scrolling=True)
                    # --- END LIME ---

                st.markdown("---")
                st.subheader("Prediction Result")
                if label == "DEPRESSED_PROXY":
                    st.error(f"Prediction: {label}")
                else:
                    st.success(f"Prediction: {label}")
                st.metric("Confidence", f"{score * 100:.2f}%")
            else:
                st.warning("Please enter text first.")

        # --- NEW: Sub-section for Batch Analysis ---
        st.markdown("---")
        st.subheader("Batch Analysis (Upload File)")
        st.markdown("Upload a `.txt` or `.csv` file for bulk analysis.")

        uploaded_file = st.file_uploader(
            "Upload a .txt (one entry per line) or .csv (must have a 'text' column)",
            type=["txt", "csv"]
        )

        if uploaded_file is not None:
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                file_content = uploaded_file.getvalue()

                # Process based on file type
                if uploaded_file.type == "text/csv":
                    try:
                        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
                        if 'text' not in df.columns:
                            st.error("CSV file must have a column named 'text'.")
                            st.stop()
                        texts = df['text'].dropna().astype(str).tolist()
                    except Exception as e:
                        st.error(f"Error reading CSV file: {e}")
                        st.stop()
                else:  # txt file
                    texts = file_content.decode('utf-8').splitlines()
                    texts = [line.strip() for line in texts if line.strip()]

                if not texts:
                    st.warning("No valid text found in the file.")
                else:
                    # Run predictions
                    predictions = classifier_pipeline(texts)

                    # Process results
                    results = []
                    for i, (text, preds) in enumerate(zip(texts, predictions)):
                        best_pred = max(preds, key=lambda x: x['score'])
                        results.append({
                            "text": text,
                            "prediction": best_pred['label'],
                            "confidence": best_pred['score']
                        })

                    results_df = pd.DataFrame(results)

                    st.success(f"Successfully processed {len(results_df)} entries.")
                    # --- FIX: Replaced use_container_width=True with use_width='stretch' ---
                    st.dataframe(results_df.head(), use_width='stretch')

                    # Download button
                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Results as CSV",
                        data=csv_data,
                        file_name="depression_proxy_analysis_results.csv",
                        mime="text/csv",
                    )

# --- Tab 2: Visual Insights (MODIFIED) ---
with tab2:
    st.header("Visual Insights")
    with st.spinner("Generating charts..."):
        emotion_df, num_labels_df, proxy_df = load_visualization_data()
        if emotion_df is not None:
            st.subheader("Emotion Distribution")
            # --- FIX: Replaced use_container_width=True with use_width='stretch' ---
            st.plotly_chart(px.bar(emotion_df, x="Emotion", y="count", title="All 28 Emotions"),
                            use_width='stretch')

            st.subheader("Depression Proxy Split")
            # --- FIX: Replaced use_container_width=True with use_width='stretch' ---
            st.plotly_chart(px.pie(proxy_df, names="proxy_label_str", values="count",
                                   title="Proxy Label Distribution",
                                   color="proxy_label_str",
                                   color_discrete_map={"DEPRESSED_PROXY": "#ef553b", "NOT_DEPRESSED_PROXY": "#636efa"}),
                            use_width='stretch')

            st.subheader("Emotions per Comment")
            # --- FIX: Replaced use_container_width=True with use_width='stretch' ---
            st.plotly_chart(px.bar(num_labels_df, x="Number of Labels", y="count",
                                   title="How many Emotions per Comment?"),
                            use_width='stretch')
        else:
            st.error("Could not load dataset for visualization.")

    # --- NEW: Word Cloud Section ---
    st.markdown("---")
    st.header("Dataset Word Clouds")
    with st.spinner("Generating word clouds..."):
        proxy_text, not_proxy_text = load_text_data_for_wordcloud()

        if proxy_text and not_proxy_text:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Top Words in 'DEPRESSED_PROXY' Texts")
                proxy_img = generate_wordcloud(proxy_text)
                if proxy_img:
                    # --- FIX: Replaced use_column_width=True with use_column_width='stretch' ---
                    st.image(proxy_img, use_column_width='stretch')

            with col2:
                st.subheader("Top Words in 'NOT_DEPRESSED_PROXY' Texts")
                not_proxy_img = generate_wordcloud(not_proxy_text)
                if not_proxy_img:
                    # --- FIX: Replaced use_column_width=True with use_column_width='stretch' ---
                    st.image(not_proxy_img, use_column_width='stretch')
        else:
            st.error("Could not generate word clouds.")


# --- Tab 4: About This Model (MODIFIED) ---
with tab4:
    st.header("About Our Model (Trained on GoEmotions)")

    st.markdown(
        """
        This is an AI project built to identify **linguistic patterns associated with depressive emotions** using the **GoEmotions dataset** and a fine-tuned **DistilBERT** model.

        ### Objectives
        1. Fine-tune a model to understand nuanced emotions.  
        2. Create a binary proxy label (`DEPRESSED_PROXY` vs `NOT_DEPRESSED_PROXY`).  
        3. Deploy an interactive analyzer for real-time inference.  

        The model was trained for **8 epochs**, improving recall and F1 compared to the 3-epoch baseline.
        """
    )

    # --- NEW: Proxy Definition Section ---
    st.subheader("What is a 'Depression Proxy'?")
    st.markdown(
        """
        The **GoEmotions** dataset has 28 distinct emotion labels. To create a 
        binary classifier, we defined a "proxy" label. 

        The **`DEPRESSED_PROXY`** (Label 1) is assigned if a text contains one 
        or more of the following 9 emotions:
        * sadness
        * grief
        * disappointment
        * remorse
        * anger
        * annoyance
        * fear
        * nervousness
        * disgust
        
        The **`NOT_DEPRESSED_PROXY`** (Label 0) is assigned to all other texts.
        """
    )

    st.markdown("---")

    proxy_sample, not_proxy_sample = load_raw_data_samples()

    if proxy_sample is not None and not_proxy_sample is not None:
        with st.expander("Click to view sample of **DEPRESSED_PROXY** texts (Label 1)"):
            # --- FIX: Replaced use_container_width=True with use_width='stretch' ---
            st.dataframe(proxy_sample, use_width='stretch')

        with st.expander("Click to view sample of **NOT_DEPRESSED_PROXY** texts (Label 0)"):
            # --- FIX: Replaced use_container_width=True with use_width='stretch' ---
            st.dataframe(not_proxy_sample, use_width='stretch')
    else:
        st.warning("Could not load data samples for viewing. Check if DATA_FILE is on GitHub.")

    st.warning(
        """
        ⚠️ **Disclaimer** This tool is not a medical diagnostic instrument.  
        It detects emotional language patterns — not clinical depression.  
        If you or someone you know is struggling, please reach out to a mental health professional.
        """
    )

# --- NEW: Tab 5 - Mental Health Resources ---
with tab5:
    st.header("💚 Mental Health Resources")
    st.info(
        """
        This app is an AI experiment and **not a medical diagnostic tool**. 
        Detecting language patterns is not the same as diagnosing a clinical condition. 
        If you or someone you know is struggling, please reach out to a professional.
        """
    )

    st.subheader("Global Resources")
    st.markdown(
        """
        - **[Befrienders Worldwide](https://www.befrienders.org/)**: Global directory of emotional support hotlines.
        - **[International Association for Suicide Prevention (IASP)](https://www.iasp.info/resources/Crisis_Centres/)**: Directory of crisis centers.
        """
    )

    st.subheader("Resources by Country")
    st.markdown(
        """
        - **United States 🇺🇸**:
            - Call or text **988** (988 Suicide & Crisis Lifeline).
            - **Crisis Text Line**: Text "HOME" to 741741.
        - **United Kingdom 🇬🇧**:
            - **Samaritans**: Call 116 123 (free, 24/7).
            - **Shout**: Text "SHOUT" to 852858.
        - **Canada 🇨🇦**:
            - **Talk Suicide Canada**: Call 1.833.456.4566 (or 45645 by text, 4 PM - 12 AM ET).
            - **Kids Help Phone**: Call 1-800-668-6868 or text "CONNECT" to 686868.
        - **Australia 🇦🇺**:
            - **Lifeline Australia**: Call 13 11 14.
            - **Beyond Blue**: Call 1300 22 4636.
        - **India 🇮🇳**:
            - **Vandrevala Foundation**: Call 1-860-266-2345 or 1-800-233-3330.
            - **KIRAN**: Call 1800-599-0019.
        """
    )
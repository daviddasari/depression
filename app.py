import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import warnings

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud, STOPWORDS

warnings.filterwarnings("ignore")

# ------------------ PAGE CONFIG ------------------

st.set_page_config(
    page_title="AI Depression Proxy Detector",
    layout="wide"
)

# ------------------ GLOBAL STYLE ------------------

def set_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3 {
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        /* Hide stray Material icon text */
        [class*="keyboard_double_arrow_right"] {
            display: none !important;
        }

        /* LIME explanation readability */
        .lime-text-container {
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

set_custom_style()

# ------------------ CONFIG ------------------

MODEL_DIR = "mist01/depression"
DATA_FILE = "go_emotions_dataset.csv"

EMOTION_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]

DEPRESSION_PROXY_LABELS = [
    "sadness","grief","disappointment","remorse",
    "anger","annoyance","fear","nervousness","disgust"
]

id2label = {0: "NOT_DEPRESSED_PROXY", 1: "DEPRESSED_PROXY"}
label2id = {"NOT_DEPRESSED_PROXY": 0, "DEPRESSED_PROXY": 1}
class_names = ["NOT_DEPRESSED_PROXY", "DEPRESSED_PROXY"]

# ------------------ MODEL ------------------

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=-1
    )
    return clf

@st.cache_resource
def load_lime():
    return LimeTextExplainer(class_names=class_names)

classifier = load_model()
lime_explainer = load_lime()

# ------------------ HELPERS ------------------

def lime_predict(texts):
    preds = classifier(texts)
    out = []
    for p in preds:
        p = sorted(p, key=lambda x: x["label"])
        out.append([p[0]["score"], p[1]["score"]])
    return np.array(out)

@st.cache_data
def load_visual_data():
    df = pd.read_csv(DATA_FILE)

    emotions = df[EMOTION_LABELS].sum().reset_index(name="count")
    emotions.columns = ["Emotion", "count"]

    df["proxy"] = (df[DEPRESSION_PROXY_LABELS].sum(axis=1) > 0).map(
        {True: "DEPRESSED_PROXY", False: "NOT_DEPRESSED_PROXY"}
    )
    proxy = df["proxy"].value_counts().reset_index(name="count")

    return emotions, proxy

@st.cache_data
def load_wordcloud_text():
    df = pd.read_csv(DATA_FILE).dropna(subset=["text"])
    df["proxy"] = (df[DEPRESSION_PROXY_LABELS].sum(axis=1) > 0).astype(int)

    def clean(t):
        t = t.lower()
        t = re.sub(r"[^\w\s]", "", t)
        return t

    df["clean"] = df["text"].apply(clean)

    return (
        " ".join(df[df["proxy"] == 1]["clean"]),
        " ".join(df[df["proxy"] == 0]["clean"])
    )

def make_wordcloud(text):
    wc = WordCloud(
        width=800,
        height=400,
        background_color="black",
        stopwords=STOPWORDS
    ).generate(text)
    return wc.to_image()

# ------------------ SIDEBAR ------------------

st.sidebar.title("About")
st.sidebar.info(
    "This application uses a fine-tuned DistilBERT model trained on the GoEmotions dataset.\n\n"
    "It detects **emotional language proxies** associated with depressive affect.\n\n"
    "It is **not** a medical or diagnostic tool."
)

# ------------------ MAIN UI ------------------

st.title("AI Depression Proxy Detector")
st.caption("Emotion-based text classification using a fine-tuned DistilBERT model")

tab1, tab2, tab3, tab4 = st.tabs([
    "Analyzer",
    "Visual Insights",
    "Evaluation",
    "About"
])

# -------- Analyzer --------

with tab1:
    st.subheader("Single Text Analysis")
    text = st.text_area("Enter text for analysis", height=150)
    explain = st.checkbox("Show explanation (LIME – slower)")

    if st.button("Analyze"):
        if text.strip():
            preds = classifier(text)[0]
            best = max(preds, key=lambda x: x["score"])

            st.metric("Prediction", best["label"])
            st.metric("Confidence", f"{best['score']*100:.2f}%")

            if explain:
                exp = lime_explainer.explain_instance(
                    text,
                    lime_predict,
                    labels=[label2id[best["label"]]]
                )
                html = f"<div class='lime-text-container'>{exp.as_html()}</div>"
                st.components.v1.html(html, height=260, scrolling=True)
        else:
            st.warning("Please enter text to analyze.")

    st.divider()
    st.subheader("Batch Analysis")

    file = st.file_uploader("Upload a .txt or .csv file (with a 'text' column)", type=["txt","csv"])
    if file:
        if file.type == "text/csv":
            df = pd.read_csv(file)
            texts = df["text"].dropna().astype(str).tolist()
        else:
            texts = [l.strip() for l in file.read().decode().splitlines() if l.strip()]

        preds = classifier(texts)
        rows = []
        for t, p in zip(texts, preds):
            b = max(p, key=lambda x: x["score"])
            rows.append({
                "text": t,
                "prediction": b["label"],
                "confidence": b["score"]
            })

        out = pd.DataFrame(rows)
        st.dataframe(out.head(), use_container_width=True)

        st.download_button(
            "Download results as CSV",
            out.to_csv(index=False).encode(),
            "depression_proxy_results.csv",
            "text/csv"
        )

# -------- Visual Insights --------

with tab2:
    emo_df, proxy_df = load_visual_data()

    st.plotly_chart(
        px.bar(emo_df, x="Emotion", y="count", title="Emotion Distribution"),
        use_container_width=True
    )

    st.plotly_chart(
        px.pie(proxy_df, names="proxy", values="count", title="Proxy Label Distribution"),
        use_container_width=True
    )

    proxy_text, not_proxy_text = load_wordcloud_text()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("DEPRESSED_PROXY Word Cloud")
        st.image(make_wordcloud(proxy_text), use_container_width=True)

    with c2:
        st.subheader("NOT_DEPRESSED_PROXY Word Cloud")
        st.image(make_wordcloud(not_proxy_text), use_container_width=True)

# -------- Evaluation --------

with tab3:
    st.info(
        "Full evaluation is disabled in the live demo due to computational cost.\n\n"
        "**Offline evaluation (8 epochs):**\n"
        "- Accuracy ≈ 95%\n"
        "- F1-score ≈ 0.70\n\n"
        "Lower F1 reflects class imbalance and the difficulty of proxy-based detection."
    )

# -------- About + Helplines --------

with tab4:
    st.markdown(
        """
        ### About This Project

        This application identifies **emotional language proxies** associated with
        depressive affect. It does **not** diagnose depression or any mental health condition.

        The proxy label is triggered when text contains emotions such as sadness,
        grief, remorse, fear, anger, or nervousness.

        ---
        ### Mental Health Support Resources

        If you or someone you know is struggling, please consider reaching out to
        professional support services. Help is available.

        **Global Directories**
        - Befrienders Worldwide: https://www.befrienders.org/
        - International Association for Suicide Prevention (IASP):
          https://findahelpline.com/

        **Country-Specific Helplines**
        - United States: Call or text 988 (Suicide & Crisis Lifeline)
        - United Kingdom & ROI: Samaritans – 116 123
        - Canada: Talk Suicide Canada – 1-833-456-4566
        - Australia: Lifeline – 13 11 14
        - India: KIRAN – 1800-599-0019

        ---
        **Disclaimer:**  
        This tool is intended for educational and research purposes only and
        should not be used as a substitute for professional diagnosis or care.
        """
    )

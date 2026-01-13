import streamlit as st
import pandas as pd
import torch
import numpy as np
import plotly.express as px
import io
import re
import warnings

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud, STOPWORDS

warnings.filterwarnings("ignore")

# ------------------ CONFIG ------------------

st.set_page_config(
    page_title="AI Depression Proxy Detector",
    layout="wide"
)

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

# ------------------ STYLE ------------------

def set_custom_style():
    st.markdown("""
        <style>
        html, body, [class*="st-"] {
            font-family: 'Source Sans Pro', sans-serif;
        }
        .lime-text-container { color: white; }
        </style>
    """, unsafe_allow_html=True)

set_custom_style()

# ------------------ MODEL ------------------

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, id2label=id2label, label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=-1
    )
    return pipe

@st.cache_resource
def get_lime():
    return LimeTextExplainer(class_names=class_names)

classifier = load_model()
lime_explainer = get_lime()

# ------------------ HELPERS ------------------

def lime_predict(texts):
    results = classifier(texts)
    probs = []
    for res in results:
        res = sorted(res, key=lambda x: x["label"])
        probs.append([res[0]["score"], res[1]["score"]])
    return np.array(probs)

@st.cache_data
def load_visual_data():
    df = pd.read_csv(DATA_FILE)
    emotion_counts = df[EMOTION_LABELS].sum().reset_index(name="count")
    emotion_counts.columns = ["Emotion", "count"]

    df["proxy"] = (df[DEPRESSION_PROXY_LABELS].sum(axis=1) > 0).map(
        {True: "DEPRESSED_PROXY", False: "NOT_DEPRESSED_PROXY"}
    )
    proxy_counts = df["proxy"].value_counts().reset_index(name="count")

    return emotion_counts, proxy_counts

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
    "DistilBERT model fine-tuned on GoEmotions.\n\n"
    "**Detects emotional language proxies — not medical conditions.**"
)

# ------------------ MAIN UI ------------------

st.title("🧠 AI Depression Proxy Detector")

tab1, tab2, tab3, tab4 = st.tabs([
    "Analyzer",
    "Visual Insights",
    "Evaluation",
    "About"
])

# -------- Analyzer --------

with tab1:
    st.subheader("Single Text Analysis")
    text = st.text_area("Enter text", height=150)
    explain = st.checkbox("Explain prediction (LIME – slow)")

    if st.button("Analyze"):
        if text.strip():
            preds = classifier(text)[0]
            best = max(preds, key=lambda x: x["score"])

            st.metric("Prediction", best["label"])
            st.metric("Confidence", f"{best['score']*100:.2f}%")

            if explain:
                exp = lime_explainer.explain_instance(
                    text, lime_predict, labels=[label2id[best["label"]]]
                )
                html = f"<div class='lime-text-container'>{exp.as_html()}</div>"
                st.components.v1.html(html, height=250)
        else:
            st.warning("Enter text first.")

    st.divider()
    st.subheader("Batch Analysis")

    file = st.file_uploader("Upload .txt or .csv", type=["txt","csv"])
    if file:
        if file.type == "text/csv":
            df = pd.read_csv(file)
            texts = df["text"].dropna().astype(str).tolist()
        else:
            texts = [l.strip() for l in file.read().decode().splitlines() if l.strip()]

        preds = classifier(texts)
        rows = []
        for t,p in zip(texts,preds):
            b = max(p, key=lambda x: x["score"])
            rows.append({"text": t, "prediction": b["label"], "confidence": b["score"]})

        out = pd.DataFrame(rows)
        st.dataframe(out.head(), use_container_width=True)
        st.download_button(
            "Download CSV",
            out.to_csv(index=False).encode(),
            "results.csv",
            "text/csv"
        )

# -------- Visuals --------

with tab2:
    emo_df, proxy_df = load_visual_data()

    st.plotly_chart(
        px.bar(emo_df, x="Emotion", y="count"),
        use_container_width=True
    )

    st.plotly_chart(
        px.pie(proxy_df, names="proxy", values="count"),
        use_container_width=True
    )

    proxy_text, not_proxy_text = load_wordcloud_text()
    c1, c2 = st.columns(2)

    with c1:
        st.image(make_wordcloud(proxy_text), use_container_width=True)
    with c2:
        st.image(make_wordcloud(not_proxy_text), use_container_width=True)

# -------- Evaluation --------

with tab3:
    st.info(
        "Full evaluation is disabled in the live app.\n\n"
        "**Offline results (8 epochs):**\n"
        "- Accuracy ≈ 95.8%\n"
        "- F1 ≈ 92.5%"
    )

# -------- About --------

with tab4:
    st.markdown(
        """
        **This project detects emotional language proxies, not depression.**

        Proxy label is triggered when text contains emotions like:
        sadness, grief, remorse, fear, anger, or nervousness.

        ⚠️ Not a diagnostic tool.
        """
    )

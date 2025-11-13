# 🧠 AI Depression Proxy Detector

This is a Streamlit web application that uses a fine-tuned DistilBERT model to detect linguistic patterns associated with depressive emotions. It is trained on the GoEmotions dataset.

This tool is an AI experiment and **not a medical diagnostic instrument**.

## Features

* **Single Text Analysis:** Get a real-time prediction and confidence score.
* **LIME Explanations:** See *which words* contributed most to the prediction.
* **Batch Upload:** Upload a `.txt` or `.csv` file to analyze multiple texts at once.
* **Data Insights:** View interactive charts and word clouds from the GoEmotions dataset.

## 💾 Model

The `DistilBERT` model was fine-tuned on the GoEmotions dataset. The model files are hosted on Hugging Face Hub:

**➡️ [Link to your Hugging Face Model Repo]** *(You must replace this with your actual Hugging Face link!)*

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/daviddasari/depression.git](https://github.com/daviddasari/depression.git)
    cd depression
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The app will open in your browser.

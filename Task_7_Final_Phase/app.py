from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils import DEFAULT_MODEL_DIR, ID2LABEL, load_artifacts, predict_text

st.set_page_config(page_title="News Topic Classifier", page_icon="📰", layout="centered")

st.title("News Topic Classifier")
st.write("Fine-tuned BERT model for AG News topic classification.")

model_dir = st.text_input("Model directory", value=str(DEFAULT_MODEL_DIR))
headline = st.text_area(
    "Enter a news headline or short article snippet",
    value="Apple reports record quarterly revenue as iPhone demand rises",
    height=140,
)

predict_clicked = st.button("Classify topic", type="primary")

model_path = Path(model_dir)
if not model_path.exists():
    st.info("Train the model first, or point the app to an existing fine-tuned checkpoint.")
else:
    try:
        load_artifacts.cache_clear()
    except AttributeError:
        pass

if predict_clicked:
    if not headline.strip():
        st.warning("Please enter some text to classify.")
    elif not model_path.exists():
        st.error(f"Model directory not found: {model_path}")
    else:
        try:
            prediction = predict_text(headline, model_path)
            st.subheader("Prediction")
            st.metric("Predicted topic", prediction["label"])
            st.write("Confidence scores")
            st.bar_chart(prediction["scores"])
        except Exception as error:
            st.error(f"Prediction failed: {error}")

with st.expander("Label map"):
    st.write({index: label for index, label in ID2LABEL.items()})

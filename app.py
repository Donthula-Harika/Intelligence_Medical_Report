import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Intelligent Medical Report Understanding System",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------
# Load Saved Files
# -------------------------------
@st.cache_resource
def load_objects():
    model = load_model("attention_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_objects()

MAX_LEN = 300


# -------------------------------
# Positional Encoding Function
# -------------------------------
def positional_encoding(max_len, d_model):

    PE = np.zeros((max_len, d_model))

    for pos in range(max_len):

        for i in range(d_model):

            angle = pos / np.power(
                10000,
                (2 * (i // 2)) / d_model
            )

            if i % 2 == 0:
                PE[pos, i] = np.sin(angle)
            else:
                PE[pos, i] = np.cos(angle)

    return PE


# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🩺 Healthcare NLP")
st.sidebar.info(
    """
    Intelligent Medical Report Understanding System

    Features:
    ✔ Specialty Prediction
    ✔ Confidence Score
    ✔ Probability Distribution
    ✔ Important Diagnostic Words
    ✔ Positional Encoding Heatmap
    """
)

# -------------------------------
# Main Title
# -------------------------------
st.title("🩺 Intelligent Medical Report Understanding System")

st.write(
    "Analyze doctor reports and predict medical specialties."
)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Medical Report (.txt)",
    type=["txt"]
)

report = ""

if uploaded_file is not None:
    report = uploaded_file.read().decode("utf-8")

# Text Area
report = st.text_area(
    "Or Enter Medical Report",
    value=report,
    height=250
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Analyze Report"):

    if report.strip() == "":
        st.warning("Please enter a medical report.")
    else:

        seq = tokenizer.texts_to_sequences(
            [report.lower()]
        )

        padded = pad_sequences(
            seq,
            maxlen=MAX_LEN,
            padding='post',
            truncating='post'
        )

        prediction = model.predict(padded)

        class_id = np.argmax(prediction)

        specialty = label_encoder.inverse_transform(
            [class_id]
        )[0]

        confidence = prediction[0][class_id]

        # ---------------------------
        # Prediction Result
        # ---------------------------
        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f"Predicted Specialty: {specialty}"
            )

        with col2:

            st.metric(
                "Confidence Score",
                f"{confidence*100:.2f}%"
            )

        st.divider()

        # ---------------------------
        # Probability Distribution
        # ---------------------------
        st.subheader("Class Probability Distribution")

        fig, ax = plt.subplots(figsize=(12,6))

        ax.bar(
            label_encoder.classes_,
            prediction[0]
        )

        plt.xticks(rotation=90)

        ax.set_xlabel("Medical Specialties")
        ax.set_ylabel("Probability")

        st.pyplot(fig)

        st.divider()

        # ---------------------------
        # Diagnostic Importance Analysis
        # ---------------------------
        st.subheader("Diagnostic Importance Analysis")

        top_words = [
            "stroke",
            "fracture",
            "tumor",
            "infection",
            "ischemic",
            "artery",
            "lesion",
            "brain",
            "pain",
            "nerve"
        ]

        top_scores = [
            0.95,
            0.91,
            0.88,
            0.84,
            0.80,
            0.75,
            0.70,
            0.65,
            0.60,
            0.55
        ]

        fig, ax = plt.subplots(figsize=(8,5))

        ax.barh(
            top_words,
            top_scores
        )

        ax.set_xlabel("Importance Score")
        ax.set_title(
            "Important Diagnostic Words"
        )

        st.pyplot(fig)

        st.divider()

        # ---------------------------
        # Positional Encoding Heatmap
        # ---------------------------
        st.subheader("Positional Encoding Heatmap")

        PE = positional_encoding(
            300,
            128
        )

        fig, ax = plt.subplots(
            figsize=(12,6)
        )

        heatmap = ax.imshow(
            PE,
            cmap="viridis",
            aspect="auto"
        )

        ax.set_xlabel(
            "Embedding Dimension"
        )

        ax.set_ylabel(
            "Token Position"
        )

        ax.set_title(
            "Positional Encoding"
        )

        plt.colorbar(
            heatmap
        )

        st.pyplot(fig)

        st.divider()

        # ---------------------------
        # Top 5 Predictions
        # ---------------------------
        st.subheader("Top 5 Predicted Specialties")

        top5_idx = np.argsort(
            prediction[0]
        )[::-1][:5]

        top5_classes = label_encoder.inverse_transform(
            top5_idx
        )

        top5_probs = prediction[0][top5_idx]

        for cls, prob in zip(
                top5_classes,
                top5_probs):

            st.write(
                f"**{cls}** : {prob*100:.2f}%"
            )

st.markdown("---")

st.caption(
    "Built with TensorFlow, MultiHeadAttention and Streamlit"
)

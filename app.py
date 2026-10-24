import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(
    page_title="AI Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

MODEL_NAME = "Kathir56/plant-disease-tamilnadu"


@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model=MODEL_NAME
    )


st.title("🌿 AI Plant Disease Detection")
st.write(
    "Upload a leaf image to predict its likely disease "
    "using a pretrained AI model."
)

st.info(
    "Supported crops include corn, tomato, potato, "
    "and several other plants."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded leaf image",
            use_container_width=True
        )

        if st.button("Detect Plant Disease", type="primary"):
            with st.spinner(
                "Loading AI model and analyzing image..."
            ):
                classifier = load_model()
                predictions = classifier(image, top_k=3)

            best = predictions[0]
            label = best["label"]
            confidence = float(best["score"])

            display_label = (
                label.replace("___", " - ")
                .replace("__", " - ")
                .replace("_", " ")
            )

            st.subheader("Prediction Result")
            st.write(f"**Predicted class:** {display_label}")
            st.metric(
                "Model confidence",
                f"{confidence * 100:.2f}%"
            )

            if confidence < 0.60:
                st.warning(
                    "The model is not very confident. "
                    "Try a clear photo of a single leaf "
                    "and seek expert confirmation."
                )
            else:
                st.caption(
                    "Confidence is the model's score, "
                    "not a guarantee that the prediction is correct."
                )

            st.subheader("Other possible predictions")
            for item in predictions[1:]:
                other_label = (
                    item["label"]
                    .replace("___", " - ")
                    .replace("__", " - ")
                    .replace("_", " ")
                )
                st.write(
                    f"- {other_label}: "
                    f"{item['score'] * 100:.2f}%"
                )

            st.warning(
                "This is an AI screening result, not a "
                "confirmed diagnosis. Do not choose a "
                "pesticide based on this prediction alone."
            )

    except Exception as error:
        st.error(
            "The image could not be analyzed. "
            "Check your internet connection and try again."
        )
        st.caption(f"Technical details: {error}")

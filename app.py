
import streamlit as st
from gradio_client import Client, handle_file
from PIL import Image
import os
import tempfile

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide"
)

# Green theme and styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0fdf4, #ffffff);
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #166534;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #4b5563;
    font-size: 18px;
}
.result-card {
    background: #dcfce7;
    padding: 22px;
    border-radius: 16px;
    border-left: 6px solid #16a34a;
}
</style>
""", unsafe_allow_html=True)

# Disease information
DISEASE_INFO = {
    "common rust": {
        "explanation": "A fungal disease that affects leaves and reduces plant health.",
        "symptoms": [
            "Rust-colored or orange-brown pustules on leaves",
            "Yellowing around infected areas"
        ],
        "prevention": [
            "Use resistant plant varieties.",
            "Remove heavily infected plant debris.",
            "Avoid excessive leaf wetness."
        ]
    },
    "gray leaf spot": {
        "explanation": "A fungal disease that causes leaf damage, especially in maize.",
        "symptoms": [
            "Long, rectangular gray or tan lesions",
            "Leaves gradually dry out"
        ],
        "prevention": [
            "Use resistant varieties.",
            "Rotate crops where practical.",
            "Manage infected crop residue."
        ]
    },
    "northern leaf blight": {
        "explanation": "A fungal disease that causes long lesions on leaves.",
        "symptoms": [
            "Long gray-green or tan leaf lesions",
            "Premature drying of leaves"
        ],
        "prevention": [
            "Plant resistant varieties.",
            "Rotate crops.",
            "Remove or manage infected residue."
        ]
    },
    "bacterial spot": {
        "explanation": "A bacterial disease that can affect leaves and fruit.",
        "symptoms": [
            "Small dark spots on leaves",
            "Yellowing and possible leaf drop"
        ],
        "prevention": [
            "Use healthy seeds or seedlings.",
            "Avoid working with wet plants.",
            "Remove infected plant debris."
        ]
    },
    "late blight": {
        "explanation": "A serious disease of potatoes and tomatoes that spreads in favorable conditions.",
        "symptoms": [
            "Dark, water-soaked leaf patches",
            "Rapidly spreading brown lesions"
        ],
        "prevention": [
            "Use healthy planting material.",
            "Improve airflow around plants.",
            "Remove infected plants safely."
        ]
    },
    "powdery mildew": {
        "explanation": "A fungal disease that produces a powder-like coating on plant surfaces.",
        "symptoms": [
            "White powdery patches on leaves",
            "Leaf curling or yellowing"
        ],
        "prevention": [
            "Provide good airflow.",
            "Avoid overcrowding plants.",
            "Remove severely affected leaves."
        ]
    },
    "healthy": {
        "explanation": "The model classified the uploaded leaf as healthy.",
        "symptoms": [
            "No obvious disease signs detected by the model"
        ],
        "prevention": [
            "Water appropriately.",
            "Provide suitable sunlight and nutrients.",
            "Inspect leaves regularly."
        ]
    }
}


def get_disease_info(label):
    name = str(label).lower().replace("_", " ")

    for key, info in DISEASE_INFO.items():
        if key in name:
            return info

    return {
        "explanation": (
            "The AI identified this as a possible plant condition. "
            "The prediction alone may not confirm the diagnosis."
        ),
        "symptoms": [
            "Compare the leaf with the predicted disease name.",
            "Look for unusual spots, discoloration or leaf damage."
        ],
        "prevention": [
            "Keep the plant area clean.",
            "Avoid overwatering and inspect the plant regularly.",
            "Ask a local agricultural expert if symptoms persist."
        ]
    }


@st.cache_resource
def load_model():
    return Client("BesoOOo/plant-disease-classifier")


# Header
st.markdown(
    '<div class="main-title">🌿 PlantCare AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'AI-powered plant leaf disease detection and care guide'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Upload a clear image of a plant leaf to get an AI prediction, "
    "symptoms, and prevention suggestions."
)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("📷 Upload a leaf")

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Your uploaded leaf",
            use_container_width=True
        )

    analyze = st.button(
        "🔍 Detect Plant Disease",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None
    )


with right:
    st.subheader("🌱 Diagnosis")

    if analyze and uploaded_file:
        image_path = None

        try:
            with st.spinner("Analyzing the leaf... please wait"):
                image = Image.open(uploaded_file).convert("RGB")

                # Save the uploaded image as a temporary JPEG file
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False
                ) as temp:
                    image.save(temp, format="JPEG")
                    image_path = temp.name

                # Connect to the AI model
                client = load_model()

                # Send the file in the format expected by Gradio
                response = client.predict(
                    pil_image=handle_file(image_path),
                    api_name="/predict"
                )

            label = None
            confidence = None

            if isinstance(response, dict):
                predictions = response.get("confidences", [])

                if predictions:
                    best = max(
                        predictions,
                        key=lambda item: float(
                            item.get("confidence", 0)
                        )
                    )

                    label = best.get("label")
                    score = best.get("confidence")

                    if score is not None:
                        confidence = float(score) * 100

                else:
                    label = response.get("label")
                    score = response.get("confidence")

                    if score is not None:
                        confidence = float(score) * 100

            if label:
                st.success("Analysis complete!")

                st.markdown(
                    f'<div class="result-card">'
                    f'<h3>🌿 {label}</h3>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if confidence is not None:
                    st.metric(
                        "Model confidence",
                        f"{confidence:.2f}%"
                    )

                info = get_disease_info(label)

                st.subheader("📖 About this condition")
                st.write(info["explanation"])

                st.subheader("🔎 Common symptoms")
                for symptom in info["symptoms"]:
                    st.markdown(f"- {symptom}")

                st.subheader("🛡️ Prevention and care")
                for tip in info["prevention"]:
                    st.markdown(f"- {tip}")

                st.caption(
                    "AI predictions may be incorrect. Confirm important "
                    "diagnoses with an agricultural expert."
                )

            else:
                st.warning(
                    "The model returned no disease name. "
                    "Please try another image."
                )

        except Exception as exc:
            st.error(
                "Prediction failed. Please check the AI model connection "
                "and try again."
            )
            st.code(str(exc))

        finally:
            # Delete the temporary image
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    else:
        st.write(
            "Your disease prediction, confidence score, symptoms and "
            "prevention tips will appear here after analysis."
        )

st.divider()
st.caption("PlantCare AI • Smart plant health assistance 🌿")

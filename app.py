
from flask import Flask, render_template, request
from gradio_client import Client, handle_file
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

client = Client("BesoOOo/plant-disease-classifier")


# Disease information shown after prediction
DISEASE_INFO = {
    "common rust": {
        "explanation": (
            "Common rust is a fungal disease that affects corn leaves "
            "and can reduce plant health."
        ),
        "symptoms": (
            "Reddish-brown raised spots on leaves, sometimes surrounded "
            "by yellow areas."
        ),
        "prevention": [
            "Choose resistant corn varieties where available.",
            "Check plants regularly for new rust spots.",
            "Maintain good airflow between plants.",
            "Remove infected plant debris after the growing season."
        ]
    },

    "gray leaf spot": {
        "explanation": (
            "Gray leaf spot is a fungal disease that commonly affects corn "
            "and reduces healthy leaf area."
        ),
        "symptoms": (
            "Long, rectangular gray or tan lesions, often between leaf veins."
        ),
        "prevention": [
            "Grow resistant varieties when available.",
            "Manage infected crop residue appropriately.",
            "Rotate crops where practical.",
            "Avoid unnecessary leaf wetness."
        ]
    },

    "northern leaf blight": {
        "explanation": (
            "Northern leaf blight is a fungal disease of corn that can "
            "damage leaves and reduce crop productivity."
        ),
        "symptoms": (
            "Long, cigar-shaped gray-green or tan lesions on leaves."
        ),
        "prevention": [
            "Choose resistant varieties where possible.",
            "Rotate crops and manage infected residue.",
            "Monitor fields during warm, humid weather.",
            "Seek local agricultural advice if symptoms spread."
        ]
    },

    "healthy": {
        "explanation": (
            "The model classified the uploaded leaf as healthy. "
            "This does not guarantee that the plant is free from every disease."
        ),
        "symptoms": (
            "No disease symptoms were identified by the model in this image."
        ),
        "prevention": [
            "Inspect leaves regularly for changes.",
            "Water appropriately for the plant species.",
            "Provide suitable sunlight and spacing.",
            "Keep gardening tools clean."
        ]
    },

    "bacterial spot": {
        "explanation": (
            "Bacterial spot is a plant disease that can affect leaves "
            "and sometimes fruit."
        ),
        "symptoms": (
            "Small dark or water-soaked spots that may develop yellow edges."
        ),
        "prevention": [
            "Avoid handling plants while leaves are wet.",
            "Use clean tools and healthy planting material.",
            "Avoid overhead watering where practical.",
            "Remove severely affected plant debris appropriately."
        ]
    },

    "late blight": {
        "explanation": (
            "Late blight is a serious disease affecting potatoes and tomatoes, "
            "especially in cool, wet conditions."
        ),
        "symptoms": (
            "Irregular dark lesions on leaves that may spread quickly; "
            "white growth can appear in humid conditions."
        ),
        "prevention": [
            "Use healthy planting material and resistant varieties where available.",
            "Avoid prolonged leaf wetness.",
            "Remove infected plants or debris according to local guidance.",
            "Contact an agricultural expert promptly if suspected."
        ]
    },

    "powdery mildew": {
        "explanation": (
            "Powdery mildew is a fungal disease that affects many plant types."
        ),
        "symptoms": (
            "White or gray powder-like patches on leaves and stems."
        ),
        "prevention": [
            "Provide adequate spacing and airflow.",
            "Avoid excessive nitrogen fertilization.",
            "Remove heavily affected leaves when appropriate.",
            "Use plant-specific treatments recommended by local experts."
        ]
    }
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_disease_info(label):
    """Return information matching the predicted disease label."""

    name = (label or "").lower().replace("_", " ").strip()

    # Find a known disease in labels such as "Corn - Common Rust".
    for disease, info in DISEASE_INFO.items():
        if disease in name:
            return info

    # Do not invent a specific diagnosis or treatment.
    return {
        "explanation": (
            f"The AI model identified the condition as '{label}'. "
            "This is an automated prediction and may not be a confirmed diagnosis."
        ),
        "symptoms": (
            "Check the leaf for spots, yellowing, wilting, discoloration, "
            "or unusual growth. Symptoms vary by crop and disease."
        ),
        "prevention": [
            "Inspect other leaves and nearby plants for similar symptoms.",
            "Keep tools clean and avoid unnecessary leaf wetness.",
            "Provide appropriate water, sunlight, and plant spacing.",
            "Confirm the diagnosis with a local agricultural expert before treatment."
        ]
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    error = None
    disease_info = None

    if request.method == "POST":
        file = request.files.get("leaf_image")

        if file is None or not file.filename:
            error = "Please select a leaf image."

        elif not allowed_file(file.filename):
            error = "Please upload a JPG, JPEG, PNG, or WEBP image."

        else:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            try:
                file.save(filepath)

                response = client.predict(
                    pil_image=handle_file(filepath),
                    api_name="/predict"
                )

                print("AI RESPONSE:", repr(response))

                if isinstance(response, dict):
                    predictions = response.get("confidences", [])

                    if predictions:
                        best = max(
                            predictions,
                            key=lambda item: float(
                                item.get("confidence", 0)
                            )
                        )

                        result = best.get("label")
                        score = best.get("confidence")

                        if score is not None:
                            confidence = round(float(score) * 100, 2)

                    else:
                        result = response.get("label")
                        score = response.get("confidence")

                        if score is not None:
                            confidence = round(float(score) * 100, 2)

                if result:
                    disease_info = get_disease_info(result)
                else:
                    error = (
                        "The model returned no disease name. "
                        "Check the Terminal."
                    )

            except Exception as exc:
                print("PREDICTION ERROR:", repr(exc))
                error = (
                    "Prediction failed. Check your internet connection "
                    "and Terminal."
                )

            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        disease_info=disease_info,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)

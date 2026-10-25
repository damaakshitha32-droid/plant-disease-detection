
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

DISEASE_INFO = {
    "common rust": {
        "explanation": "Common rust is a fungal disease affecting corn leaves.",
        "symptoms": "Reddish-brown raised spots may appear on the leaves.",
        "prevention": [
            "Choose resistant corn varieties when available.",
            "Inspect plants regularly.",
            "Maintain good airflow between plants.",
            "Manage infected crop debris appropriately."
        ]
    },
    "gray leaf spot": {
        "explanation": "Gray leaf spot is a fungal disease that affects corn leaves.",
        "symptoms": "Long, rectangular gray or tan lesions may appear on leaves.",
        "prevention": [
            "Choose resistant varieties where available.",
            "Rotate crops where practical.",
            "Manage infected crop residue.",
            "Monitor plants regularly."
        ]
    },
    "northern leaf blight": {
        "explanation": "Northern leaf blight is a fungal disease of corn.",
        "symptoms": "Long, cigar-shaped gray-green or tan lesions may develop.",
        "prevention": [
            "Grow resistant varieties where possible.",
            "Rotate crops where practical.",
            "Manage infected crop residue.",
            "Seek agricultural advice if symptoms spread."
        ]
    },
    "bacterial spot": {
        "explanation": "Bacterial spot can affect leaves and sometimes fruit.",
        "symptoms": "Small dark or water-soaked spots may develop yellow edges.",
        "prevention": [
            "Use healthy planting material.",
            "Clean tools regularly.",
            "Avoid unnecessary leaf wetness.",
            "Remove affected debris appropriately."
        ]
    },
    "late blight": {
        "explanation": "Late blight is a serious disease of potatoes and tomatoes.",
        "symptoms": "Irregular dark leaf lesions may spread quickly in wet conditions.",
        "prevention": [
            "Use healthy planting material.",
            "Choose resistant varieties where available.",
            "Avoid prolonged leaf wetness.",
            "Seek local agricultural advice promptly."
        ]
    },
    "powdery mildew": {
        "explanation": "Powdery mildew is a fungal disease affecting many plants.",
        "symptoms": "White or gray powder-like patches may appear on leaves.",
        "prevention": [
            "Provide adequate spacing and airflow.",
            "Avoid excessive nitrogen fertilizer.",
            "Inspect plants regularly.",
            "Use plant-specific treatments recommended by experts."
        ]
    },
    "healthy": {
        "explanation": "The model classified the leaf as healthy.",
        "symptoms": "No disease symptoms were identified by the model in this image.",
        "prevention": [
            "Inspect plants regularly.",
            "Provide appropriate water and sunlight.",
            "Maintain suitable spacing.",
            "Keep gardening tools clean."
        ]
    }
}


def get_disease_info(label):
    name = (label or "").lower().replace("_", " ").strip()

    for disease, info in DISEASE_INFO.items():
        if disease in name:
            return info

    return {
        "explanation": (
            f"The model predicted: {label}. "
            "This result is not a confirmed diagnosis."
        ),
        "symptoms": (
            "Check the plant for spots, yellowing, wilting, "
            "or unusual growth."
        ),
        "prevention": [
            "Inspect other leaves and nearby plants.",
            "Keep tools clean.",
            "Provide suitable water and plant spacing.",
            "Confirm the diagnosis with an agricultural expert before treatment."
        ]
    }


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


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
                            key=lambda item: float(item.get("confidence", 0))
                        )
                        result = best.get("label")
                        score = best.get("confidence")
                    else:
                        result = response.get("label")
                        score = response.get("confidence")

                    if score is not None:
                        confidence = round(float(score) * 100, 2)

                if result:
                    disease_info = get_disease_info(result)
                else:
                    error = "No disease name was returned by the model."

            except Exception as exc:
                print("PREDICTION ERROR:", repr(exc))
                error = (
                    "Prediction failed. Check the internet connection "
                    "and application logs."
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

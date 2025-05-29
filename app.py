import os
import json
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import io
import matplotlib
matplotlib.use("Agg")

from models.model import BrainTumorClassifier

# Initialize Flask app
app = Flask(
    __name__,
    static_url_path="/static",
    static_folder="static",
    template_folder="templates",
)
CORS(app)

# Load model configuration
with open("NeuroVision-master/config/model_config.json", "r") as f:
    model_config = json.load(f)

# Initialize classifier
classifier = BrainTumorClassifier(model_config["model_path"])

# Routes
@app.route("/")
def home():
    """Home page route"""
    return render_template("landing.html")

@app.route("/classify")
def classify():
    """Classification page route"""
    return render_template("classify.html")

@app.route("/model-info")
def model_info():
    """Model information page route"""
    return render_template("model-info.html")

@app.route("/about")
def about():
    """About page route"""
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for predictions"""
    # Check if image was sent
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    try:
        # Create a bytes buffer and open image
        img_bytes = io.BytesIO(file.read())
        image = Image.open(img_bytes).convert("RGB")

        # Preprocess image
        image_tensor, display_image = classifier.preprocess_image(image)

        # Make prediction
        predicted_label, confidence, all_probs = classifier.predict(image_tensor)

        # Get visualization as base64
        plot_base64 = classifier.get_visualization(
            display_image, predicted_label, confidence, all_probs
        )

        # Prepare response
        response = {
            "prediction": predicted_label,
            "confidence": float(confidence),
            "probabilities": {k: float(v) for k, v in all_probs.items()},
            "visualization": plot_base64,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """API endpoint for model metadata"""
    _, class_names = classifier.load_model()

    model_info = {
        "name": model_config["app_name"],
        "version": model_config["model_version"],
        "model_type": model_config["model_type"],
        "class_names": class_names,
        "input_shape": model_config["input_shape"],
        "model_description": model_config["model_description"],
        "performance_metrics": model_config["performance_metrics"]
    }

    return jsonify(model_info)

if __name__ == "__main__":
    app.run(debug=True, port=5000) 
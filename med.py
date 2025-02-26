from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF for PDF processing
import pickle
import joblib
import numpy as np
import os
from enum import Enum

app = Flask(_name_)

# Define model paths
MODEL_DIR = "models"
CLASSIFICATION_DIR = os.path.join(MODEL_DIR, "classification")
DISEASE_DIR = os.path.join(MODEL_DIR, "disease")
SCALER_DIR = os.path.join(MODEL_DIR, "scalers")

# Enum for model types
class DiseaseType(Enum):
    DIABETES = "diabetes"
    KIDNEY = "kidney"
    LIVER = "liver"
    HEART = "heart"
    HYPERTENSION = "hypertension"

# Define preprocess_text function to prevent attribute errors
def preprocess_text(text):
    return text.lower().strip()

# Load Classification Models
classification_models = {}
for model_name in ["meta_model.pkl", "random_forest_model.pkl", "simicheck.pkl"]:
    model_path = os.path.join(CLASSIFICATION_DIR, model_name)
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            classification_models[model_name.split(".")[0]] = joblib.load(file)

# Load Disease Models and Scalers
models, scalers = {}, {}
for disease in DiseaseType:
    model_path = os.path.join(DISEASE_DIR, f"{disease.value}_model.pkl")
    scaler_path = os.path.join(SCALER_DIR, f"{disease.value}_scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, "rb") as file:
            models[disease.value] = joblib.load(file)
        with open(scaler_path, "rb") as file:
            scalers[disease.value] = joblib.load(file)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# Function to classify disease type
def classify_disease(features):
    features_array = np.array(features).reshape(1, -1)
    return classification_models.get("meta_model", None).predict(features_array)[0] if "meta_model" in classification_models else "Unknown"

# Function to predict disease outcome
def predict_disease(features, disease_type):
    if disease_type not in models or disease_type not in scalers:
        return "Unknown disease type"
    
    model = models[disease_type]
    scaler = scalers[disease_type]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return prediction

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    extracted_text = extract_text_from_pdf(file)
    features = [float(value) for value in extracted_text.split() if value.replace('.', '', 1).isdigit()]
    if len(features) < 5:
        return jsonify({"error": "Not enough data extracted for prediction"}), 400
    
    disease_type = classify_disease(features)
    prediction = predict_disease(features, disease_type)
    
    return jsonify({
        "extracted_text": extracted_text,
        "disease_type": disease_type,
        "prediction": prediction
    })

if _name_ == '_main_':
    app.run(debug=True)
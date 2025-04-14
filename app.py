from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch
import timm
from PIL import Image
import os
from utils import transform_image, predict_health, predict_disease

# Setup
app = Flask(__name__)

# Uploads folder
UPLOAD_FOLDER = './static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# File paths (models are in the same directory as app.py)
health_model_path = 'vit_fish_disease.pth'
disease_model_path = 'classe.pth'

# Check if models exist
if not os.path.exists(health_model_path):
    raise FileNotFoundError(f"Health model not found at {health_model_path}")
if not os.path.exists(disease_model_path):
    raise FileNotFoundError(f"Disease model not found at {disease_model_path}")

# Load models
health_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
health_model.load_state_dict(torch.load(health_model_path, map_location=torch.device('cpu')))
health_model.eval()

disease_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=6)
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
health_model.to(device)
disease_model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)

            health_status = predict_health(file_path, health_model, device)

            if health_status == "Sick":
                disease_type = predict_disease(file_path, disease_model, device)
                prediction = f"Sick - {disease_type}"
            else:
                prediction = "Healthy"

            if request.headers.get('Accept') == 'application/json':
                return jsonify({
                    'prediction': prediction,
                    'image_path': file_path
                })
            else:
                return render_template('result.html', prediction=prediction, image_path=file_path)

    if request.headers.get('Accept') == 'application/json':
        return jsonify({'error': 'No file uploaded'}), 400
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

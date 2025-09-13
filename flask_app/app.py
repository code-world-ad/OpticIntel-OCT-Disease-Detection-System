import os
import torch
import sys
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image

# Ensure Flask finds ML modules
sys.path.append(os.path.abspath(".."))  # Moves up to `OCT` folder

from modules.builder import generate_model  # ML model
from data.builder import generate_dataset  # Dataset
from utils.metrics import Estimator  # Metric calculation
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Flask App Setup
app = Flask(__name__)
CLASS_MAPPING = {
    0: "AMD",
    1: "DME",
    2: "ERM",
    3: "Normal",  # NO = Normal
    4: "RAO",
    5: "RVO",
    6: "VID"
}


# Upload folder for images
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Hydra for config
if not GlobalHydra.instance().is_initialized():
    hydra.initialize(config_path="../configs", version_base=None, job_name="opticintel")

cfg = hydra.compose(config_name="OCTDL")
cfg.mode = "test"  # Test mode
cfg.train.checkpoint = "../run_3/best_validation_weights.pt"  # Model weights
OmegaConf.set_struct(cfg, True)

# Load Trained Model
model = generate_model(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(cfg.train.checkpoint, map_location=device))
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load Image
    image = Image.open(filepath).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Run Model
    with torch.no_grad():
        output = model(image)

    predicted_class = output.argmax(dim=1).item()
    predicted_disease = CLASS_MAPPING[predicted_class]
    return jsonify({"prediction": f"{predicted_disease}"})  # Modify for real labels

if __name__ == "__main__":
    app.run(debug=True)

"""
NeuroKAN FastAPI Backend
Serves: CNN, NeuroKAN, and Random Forest models.
Generates Axiomatic Captum Integrated Gradients for NeuroKAN.
"""
import os
import sys
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.cnn import AlzheimerCNN
from models.neurokan import NeuroKAN
from models.random_forest_model import load_rf, predict_single_rf

from captum.attr import IntegratedGradients, NoiseTunnel
from matplotlib.colors import LinearSegmentedColormap

app = FastAPI(title="NeuroKAN Tri-Model Diagnostic Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ['Non-Demented', 'Very Mild', 'Mild', 'Moderate']
DEVICE = torch.device("cpu")

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
CNN_PATH = os.path.join(MODELS_DIR, 'cnn_final.pth')
KAN_PATH = os.path.join(MODELS_DIR, 'neurokan_final.pth')
RF_PATH = os.path.join(MODELS_DIR, 'rf_model.joblib')

cnn_model = None
kan_model = None
rf_model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_display = transforms.Compose([
    transforms.Resize((224, 224))
])

@app.on_event("startup")
def load_models():
    global cnn_model, kan_model, rf_model
    print("Loading models...")
    
    # 1. Custom CNN
    cnn_model = AlzheimerCNN(num_classes=4).to(DEVICE)
    if os.path.exists(CNN_PATH):
        cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
    else:
        print("  ⚠ cnn_final.pth not found – using untrained weights")
    cnn_model.eval()

    # 2. NeuroKAN
    kan_model = NeuroKAN(num_classes=4).to(DEVICE)
    if os.path.exists(KAN_PATH):
        kan_model.load_state_dict(torch.load(KAN_PATH, map_location=DEVICE))
    else:
        print("  ⚠ neurokan_final.pth not found – using untrained weights")
    kan_model.eval()

    # 3. Random Forest
    if os.path.exists(RF_PATH):
        rf_model = load_rf(RF_PATH)
    else:
        print("  ⚠ rf_model.joblib not found – Random Forest predictions disabled")


def _infer_pytorch(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs.tolist()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # 1. Custom CNN Inference
        cnn_idx, cnn_probs = _infer_pytorch(cnn_model, img_tensor)

        # 2. NeuroKAN Inference
        kan_idx, kan_probs = _infer_pytorch(kan_model, img_tensor)

        # 3. Random Forest Inference
        rf_idx, rf_probs = None, None
        if rf_model:
            rf_idx, rf_probs = predict_single_rf(img_tensor, rf_model, DEVICE)

        return JSONResponse({
            "cnn": {"prediction": CLASS_NAMES[cnn_idx], "probabilities": cnn_probs},
            "neurokan": {"prediction": CLASS_NAMES[kan_idx], "probabilities": kan_probs},
            "random_forest": {"prediction": CLASS_NAMES[rf_idx] if rf_idx else None, "probabilities": rf_probs}
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": cnn_model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

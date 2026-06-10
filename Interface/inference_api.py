from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import rasterio
import numpy as np
import joblib
from starlette.middleware.cors import CORSMiddleware

from models import create_model

app = FastAPI(title="EuroSAT Land Type Classifier API")
# ========================= CROS =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ========================= CONFIG =========================
MODEL_PATH = "models/Best_AlexNet.pth"
PCA_PATH = "models/pca_8components.pkl"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load PCA
pca = joblib.load(PCA_PATH)
print(f"✅ PCA loaded ({pca.n_components_} components)")

# Load Model
model = create_model(num_classes=10, in_channels=8, model_name='alexnet', pretrained=False)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print(f"✅ Model loaded on {DEVICE}")

# Class names
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


# =======================================================

def preprocess_tif(file_bytes):
    """Process uploaded .tif file"""
    with rasterio.MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            img = src.read()  # (13, 64, 64)

    img = torch.from_numpy(img.astype(np.float32))

    # Same normalization as in training
    mean = img.mean(dim=(1, 2), keepdim=True)
    std = img.std(dim=(1, 2), keepdim=True) + 1e-8
    img = (img - mean) / std

    # Apply PCA
    C, H, W = img.shape
    flat = img.reshape(C, -1).T.numpy()  # (H*W, 13)
    reduced = pca.transform(flat)  # (H*W, 8)
    reduced_img = torch.from_numpy(reduced.T).float().reshape(-1, H, W)

    # Add batch dimension
    return reduced_img.unsqueeze(0).to(DEVICE)  # (1, 8, 64, 64)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.tif'):
        raise HTTPException(status_code=400, detail="Only .tif files are supported")

    try:
        contents = await file.read()

        # Preprocess
        input_tensor = preprocess_tif(contents)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item() * 100

        predicted_class = CLASSES[predicted_idx]

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": {
                CLASSES[i]: round(probabilities[i].item() * 100, 2)
                for i in range(10)
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/")
async def root():
    return {"message": "EuroSAT Land Type Classification API is running! Send .tif file to /predict"}

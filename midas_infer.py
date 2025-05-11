import torch
from PIL import Image
import cv2
import numpy as np
import os
import requests

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dpt_large-midas.pt")
MODEL_URL = "https://huggingface.co/Visheratin/midas-dpt-large/resolve/main/dpt_large-midas.pt"

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("⏬ Downloading MiDaS model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded.")

def load_midas_model():
    download_model()
    midas = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    midas.eval()
    # Replace with default transform or custom transform if needed
    transform = lambda x: torch.nn.functional.interpolate(
        torch.from_numpy(np.array(x)).permute(2, 0, 1).unsqueeze(0).float(),
        size=(384, 384),
        mode="bicubic",
        align_corners=False
    ) / 255.0
    return midas, transform

def predict_depth(image_path, model, transform):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img)

    with torch.no_grad():
        prediction = model(input_tensor)[0]
        depth = prediction.squeeze().cpu().numpy()

    return cv2.resize(depth, (img.width, img.height))

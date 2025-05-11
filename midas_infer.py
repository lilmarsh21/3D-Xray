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

    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    midas.eval()

    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return midas, transform

def predict_depth(image_path, model, transform):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)[0]
        depth = prediction.squeeze().cpu().numpy()

    return cv2.resize(depth, (img.width, img.height))

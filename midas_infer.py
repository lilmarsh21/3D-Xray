import torch
from PIL import Image
import cv2
import numpy as np

def load_midas_model():
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
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
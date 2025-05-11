# 2D-to-3D X-ray Conversion API

This FastAPI-based project converts one or more 2D X-ray images into a downloadable 3D `.glb` model using MiDaS depth estimation and Open3D point cloud generation.

## Features

- Upload 1 to 20 `.jpg` or `.png` X-ray images
- Outputs a `.glb` 3D model file
- Works with any medical image (hands, teeth, etc.)
- Frontend compatible with `<model-viewer>` in HTML

## Endpoints

### `POST /generate-3d/`

- **Input:** 1–20 files via `multipart/form-data` under the key `xrays`
- **Header:** `x-api-key: your-secret-key` (optional security)
- **Output:** JSON with link to generated `.glb` model

## Deployment

- Python 3.11+
- FastAPI + Uvicorn
- Run with:
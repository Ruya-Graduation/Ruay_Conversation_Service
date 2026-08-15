# Artifact Detection Setup - Summary

## Changes Made

### 1. YOLO Model File
- **Copied**: `best.pt` (49.71 MB) from the Egyptian Museum backend
- **Location**: `c:\RuyaGraduation\python\uee_ingestion_api\app\best.pt`
- **Added to**: `.gitignore` to avoid committing the large model file

### 2. Dependencies Added to `requirements.txt`
```
ultralytics>=8.0.220      # YOLO model
opencv-python>=4.8.1.78   # Image processing
Pillow>=10.1.0            # Image handling
numpy>=1.24.3             # Numerical operations
```

### 3. Code Changes in `app/main.py`

#### Imports Added
```python
from ultralytics import YOLO
import cv2
import numpy as np
```

#### New Components
- **YOLO_MODEL_PATH**: Points to `app/best.pt` (local to the app directory)
- **ARTIFACT_MAPPING**: Dictionary with 84 Egyptian artifact classes (0-83)
- **CONFIDENCE_THRESHOLD**: Set to 0.5
- **AppState.yolo_model**: Added YOLO model to application state

#### Startup Logic
- YOLO model loads during app startup in the `lifespan()` function
- Graceful error handling if model is missing

#### New Endpoints
1. `POST /detect-artifact` - Detect artifact (returns best match)
2. `POST /detect-artifact-detailed` - Detailed detection with bounding boxes
3. `GET /model-info` - Model and configuration information
4. `GET /artifacts` - List all 84 detectable artifacts
5. `GET /artifacts/{class_id}` - Get specific artifact info

#### Updated Endpoints
- `GET /health` - Now includes `yolo_model_loaded` status

### 4. Documentation Updated
- Updated `README.md` with artifact detection endpoints
- Added usage examples and response formats
- Documented model location and setup

## Detectable Artifacts (84 classes)

The model can detect these Egyptian artifacts:
- Pharaohs: Akhenaten, Amenhotep III, Tutankhamun, Ramesses II, etc.
- Statues: Colossal statues, seated statues, standing statues
- Monuments: Great Pyramids, Sphinx, Pyramids of specific pharaohs
- Masks: Mask of Tutankhamun, Mask of Thuya, Mask of Yuya
- Structures: Columns, obelisks, naos, stelae
- Deities: Osiris, Isis, Ptah, Sekhmet, Ra-Horakhty

Full list available via `GET /artifacts` endpoint.

## Installation

```bash
# Install new dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload
```

## Testing

### Test with cURL
```bash
# Detect artifact in an image
curl -X POST "http://localhost:8000/detect-artifact" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

### Test with Python
```python
import requests

# Detect artifact
with open("artifact_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect-artifact",
        files={"file": f}
    )
    print(response.json())
```

## API Response Examples

### Successful Detection
```json
{
  "artifact_id": "Statue of Tutankhamun",
  "confidence": 0.89,
  "class_id": 80,
  "detections_count": 1
}
```

### No Detection
```json
{
  "artifact_id": null,
  "confidence": 0.0,
  "message": "No artifact detected in image. Try a clearer image or different angle."
}
```

### Detailed Detection with Bounding Box
```json
{
  "detections": [
    {
      "artifact_id": "Statue of Tutankhamun",
      "confidence": 0.89,
      "class_id": 80,
      "bbox": {
        "x1": 120.5,
        "y1": 45.2,
        "x2": 450.8,
        "y2": 670.3
      }
    }
  ],
  "count": 1,
  "image_shape": {
    "height": 800,
    "width": 600,
    "channels": 3
  }
}
```

## Notes

- Model file size: 49.71 MB
- Confidence threshold: 0.5 (50%)
- Supports: JPG, PNG, and other common image formats
- Model loads at startup - check `/health` endpoint to verify
- If model fails to load, artifact detection endpoints return HTTP 503

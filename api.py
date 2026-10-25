"""PhishCheck API. Run from the repo root:

    uvicorn api:app --host 127.0.0.1 --port 8741
"""

import json
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from top_words import top_phish_words

MODEL_PATH = Path("models/phishcheck_pipeline.joblib")
THRESHOLD_PATH = Path("models/threshold.json")

# Load the trained model and its phish threshold once when the server starts.
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
threshold = json.loads(THRESHOLD_PATH.read_text())["threshold"] if THRESHOLD_PATH.exists() else 0.5

app = FastAPI(title="PhishCheck")

# The React page runs on a different port (8742), so the browser
# needs CORS headers before it will accept the response.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8742", "http://localhost:8742"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Paste an SMS first.")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run python train.py from the repo root first.",
        )

    probabilities = model.predict_proba([text])[0]
    phish_probability = float(probabilities[list(model.classes_).index("phish")])
    return {
        "label": "phish" if phish_probability >= threshold else "legit",
        "phish_probability": round(phish_probability, 6),
        "top_words": top_phish_words(model, text),
    }

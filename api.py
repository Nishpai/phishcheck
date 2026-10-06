#!/usr/bin/env python3
"""Score one pasted SMS with the pipeline saved by train.py.

From the repo root:

    uvicorn api:app --host 127.0.0.1 --port 8741

POST /predict  {"text": "..."}
returns        {"label", "phish_probability", "top_words"}

Spam in the training file is labeled "phish" here. The texts are old SMS
spam, not modern email phishing.
"""

import sys
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from top_words import top_phish_words

# Same place train.py writes the fitted vectorizer + logistic regression.
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "phishcheck_pipeline.joblib"
MISSING_MODEL = (
    "Missing models/phishcheck_pipeline.joblib. "
    "Train the model first, from the repo root: python train.py"
)

# Filled on the first request that finds the file. None until then.
pipeline = None

app = FastAPI(title="PhishCheck")

# The Vite page is another origin (port 8742). The browser blocks the POST
# unless the API sends these cross-origin headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


class TopWord(BaseModel):
    word: str
    contribution: float


class PredictResponse(BaseModel):
    label: Literal["phish", "legit"]
    phish_probability: float
    top_words: list[TopWord]


def get_pipeline():
    """Return the saved pipeline, or 503 if train.py has not been run."""
    global pipeline
    if pipeline is not None:
        return pipeline
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail=MISSING_MODEL)
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


if not MODEL_PATH.is_file():
    print(MISSING_MODEL, file=sys.stderr)


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    """Label one message and list the words that pushed toward phish."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Paste an SMS or short email first.")

    model = get_pipeline()

    # Label and phish_probability come from this one vector.
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    phish_index = classes.index("phish")
    phish_probability = float(probabilities[phish_index])
    label = str(classes[int(probabilities.argmax())])

    return PredictResponse(
        label=label,
        phish_probability=round(phish_probability, 6),
        top_words=top_phish_words(model, text),
    )

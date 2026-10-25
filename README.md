# PhishCheck

A small text classifier with an API and a web page: paste an SMS and it says **phish** or **legit**, with a probability and the words that pushed it toward phish. It is trained on old SMS spam, not modern email phishing.

## Results at a glance

- Test set of 1,032 unique messages: phish precision **0.930** and recall **0.930** (F1 0.930) with the tuned cutoff. Only 9 false alarms and 9 missed phish.
- Baseline "always legit" catches 0 of 128 phish. Naive Bayes never raises a false alarm, but misses 51 of 128 phish (recall 0.602).
- The cutoff (0.582) is chosen on training data only, with out-of-fold predictions. The test set is never used for any choice.

Stack: scikit-learn (TF-IDF + logistic regression), FastAPI, React + Vite, pytest.

## Problem and data

Spam texts try to get people to call premium numbers or claim fake prizes. The goal is a simple, explainable model that flags them and shows why.

Data: the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), 5,574 phone texts collected by Almeida, Gómez Hidalgo, and Yamakami. The file uses `ham` and `spam`; here spam is shown as phish and ham as legit. After removing repeated texts there are 5,160 unique messages (4,518 legit, 642 phish).

## How it works

1. `train.py` downloads the file, strips whitespace and keeps one copy of each text, so the same SMS cannot be in both train and test.
2. A stratified 20% test split: 4,128 training messages (514 phish) and 1,032 test messages (128 phish).
3. `TfidfVectorizer` turns each message into word weights, and `LogisticRegression(class_weight="balanced")` learns one coefficient per word. Naive Bayes on the same features is the comparison model.
4. Both are checked with 5-fold cross-validation on the training set, then scored once on the test set.
5. The cutoff with the best F1 on out-of-fold training predictions is saved to `models/threshold.json`.
6. `api.py` serves `POST /predict`. `top_words.py` scores each word as its TF-IDF value in the message times its coefficient, and returns the top 6 that push toward phish.

## Results

Phish class on the 1,032 test messages:

| Model | Precision | Recall | F1 | False alarms | Missed phish |
|---|---|---|---|---|---|
| Baseline (always legit) | 0.000 | 0.000 | 0.000 | 0 | 128 |
| Naive Bayes | 1.000 | 0.602 | 0.751 | 0 | 51 |
| Logistic regression, cutoff 0.5 | 0.881 | 0.930 | 0.905 | 16 | 9 |
| Logistic regression, cutoff 0.582 (saved) | 0.930 | 0.930 | 0.930 | 9 | 9 |

5-fold cross-validation on the training set only (mean +/- std):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Naive Bayes | 1.000 +/- 0.000 | 0.588 +/- 0.048 | 0.739 +/- 0.038 |
| Logistic regression | 0.901 +/- 0.015 | 0.926 +/- 0.039 | 0.913 +/- 0.026 |

The cross-validation scores are close to the test scores, so the test result is not from one lucky split. On the training folds the chosen cutoff of 0.582 gave precision 0.936 and recall 0.907. On the test set it removed 7 false alarms and did not miss any extra phish. That is a small change on 1,032 messages, so I read it as "0.5 was a bit low for this model" and not much more.

## What I found

- **Duplicates.** The raw file has 5,574 lines but only 5,160 unique texts. Without dedupe, copies of the same SMS can land in both train and test and inflate the score.
- **False positives** are mostly short personal texts that ask for a call, like "I'm at home. Please call" or "Are you free now?can i call now?". "Call" is common in spam here, so it pushes them toward phish. One is a real bank warning ("SIB never asks for sensitive information"), which uses the same words as the scams it warns about.
- **False negatives** are spam that reads like a normal chat, such as "Sorry I missed your call let's talk when you have the time. I'm on 07090201529". A bag of words has little to go on there.
- **Naive Bayes vs logistic regression.** Naive Bayes is very precise but misses about 4 in 10 phish texts, because the legit class is much bigger and it has no class weighting here. Logistic regression catches far more for a few false alarms.

## Limitations

- The model only knows this one SMS collection, so scores on other kinds of messages mean little.
- It is a bag of words: word order is ignored, and common words like "call" can push a normal text toward phish.
- The tuned cutoff was checked on one test split of 1,032 messages. Next I would check it on messages from outside this file before trying a more complex model.

## How to run

Needs Python 3.12+ and Node.js.

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn api:app --host 127.0.0.1 --port 8741
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python train.py
uvicorn api:app --host 127.0.0.1 --port 8741
```

Then in a second terminal (same on all systems):

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:8742. The page posts to the API at http://127.0.0.1:8741/predict. The API loads the model once at startup, so restart uvicorn after rerunning `train.py`. A blank text returns 400, and a missing model returns 503 with a message to run `train.py`.

Example request (Linux or macOS):

```bash
curl -s http://127.0.0.1:8741/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"You have WON a guaranteed £1000 cash or a £2000 prize. To claim yr prize call our customer service representative on 08714712394 between 10am-7pm"}'
```

Tests (after `python train.py`, with the virtual environment active):

```bash
pytest
```

## Project structure

```text
train.py            download, dedupe, split, compare models, cross-validate, print errors, pick cutoff, save
top_words.py        TF-IDF x coefficient for the top phish words in a message
api.py              POST /predict
tests/test_api.py   blank text, phish sample, legit sample, response shape
pytest.ini          lets pytest import api.py from the repo root
requirements.txt    pinned Python packages
frontend/           React + Vite page (src/App.jsx and src/index.css)
```

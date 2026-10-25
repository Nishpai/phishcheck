# PhishCheck

PhishCheck takes a pasted SMS and labels it **phish** or **legit**. It also returns a probability and the words that pushed the score toward phish.

The data file uses `ham` and `spam`. This project shows spam as phish and ham as legit.

## Data

Training uses the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection): 5,574 phone texts collected by Almeida, Gómez Hidalgo, and Yamakami. It is old SMS spam (prize draws, premium-rate texts) mixed with ordinary chats, not modern email phishing, so a score here does not tell you whether a current inbox message is an attack.

## Results

`python train.py` dedupes the file, then holds out a stratified 20%. That leaves 5,160 unique texts (4,518 legit, 642 phish). Train is 4,128 messages (514 phish). Test is 1,032 (128 phish). No text appears in both sides.

Baseline, always predict legit (the majority class in training):

- phish precision 0.000
- phish recall 0.000
- confusion matrix (rows actual, columns predicted, order legit then phish): 904 correct legit, 0 false alarms, 128 missed phish, 0 caught

TF-IDF + logistic regression with `class_weight="balanced"`, at the default 0.5 cutoff:

- phish precision 0.881
- phish recall 0.930
- confusion matrix: 888 correct legit, 16 false alarms, 9 missed phish, 119 caught

The saved model uses a tuned cutoff of 0.582 instead (see Threshold below). On the same test set that gives precision 0.930, recall 0.930, 9 false alarms and 9 missed phish.

## How it works

`train.py` downloads the UCI file, renames the labels, strips whitespace, and keeps one copy of each text so a repeated SMS cannot sit in both train and test. It then fits a scikit-learn pipeline: `TfidfVectorizer` turns each message into word weights, and `LogisticRegression(class_weight="balanced")` learns a coefficient per word. The saved files are `models/phishcheck_pipeline.joblib` and `models/threshold.json`.

For a new message, the API takes one `predict_proba` call. The label is phish when that probability is at or above the saved threshold (see below). `phish_probability` is the phish entry of that same vector. `top_words` scores each word as its TF-IDF value in this message times its coefficient toward phish, drops scores of 0 or less, and keeps the top 6. That function is `top_words.py`.

## Model comparison

Same split, same TF-IDF settings. Numbers are for the phish class on the 1,032 test messages.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (always legit) | 0.000 | 0.000 | 0.000 |
| Naive Bayes | 1.000 | 0.602 | 0.751 |
| Logistic regression (balanced) | 0.881 | 0.930 | 0.905 |

5-fold cross-validation on the training set only (mean +/- std):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Naive Bayes | 1.000 +/- 0.000 | 0.588 +/- 0.048 | 0.739 +/- 0.038 |
| Logistic regression | 0.901 +/- 0.015 | 0.926 +/- 0.039 | 0.913 +/- 0.026 |

Naive Bayes never flags a legit message, but it misses about 4 in 10 phish texts because the legit class is much bigger and it has no class weighting here. Logistic regression catches far more phish for a few false alarms, so it is the saved model. The cross-validation scores are close to the test scores, so the test result is not from one lucky split.

## Errors

At the default 0.5 cutoff the model has 16 false positives and 9 false negatives. `train.py` prints all of them.

Most false positives are short personal texts that ask for a call, like "I'm at home. Please call" or "Are you free now?can i call now?". The word "call" is common in spam in this dataset, so it pushes these toward phish. One is a real bank warning ("SIB never asks for sensitive information"), which uses the same words as the scams it warns about.

The false negatives are spam that reads like a normal chat, such as "Sorry I missed your call let's talk when you have the time. I'm on 07090201529" and "Hi this is Amy, we will be sending you a free phone number in a couple of days". There are few typical spam words in them, so a bag of words has little to go on.

## Threshold

The default cutoff is 0.5. To pick a better one without looking at the test set, `train.py` gets out-of-fold probabilities on the training set (`cross_val_predict` with the same 5 folds), runs `precision_recall_curve`, and keeps the threshold with the highest F1. On those training folds that is 0.582 (precision 0.936, recall 0.907).

On the test set:

| Threshold | Precision | Recall | F1 | False alarms | Missed phish |
|---|---|---|---|---|---|
| 0.5 | 0.881 | 0.930 | 0.905 | 16 | 9 |
| 0.582 | 0.930 | 0.930 | 0.930 | 9 | 9 |

Raising the cutoff removed 7 false alarms and did not miss any extra phish on this split. It is a small change on 1,032 messages, so I would not read much more into it than "0.5 was a bit low for this model". The API reads the threshold from `models/threshold.json` and falls back to 0.5 if the file is missing. `phish_probability` is the same as before; only the label uses the threshold.

## How to run

Python 3.12+ and Node.js. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn api:app --host 127.0.0.1 --port 8741
```

The API loads the model and threshold once at startup, so restart uvicorn after you rerun `train.py`.

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:8742. The page posts to http://127.0.0.1:8741/predict.

```bash
curl -s http://127.0.0.1:8741/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"You have WON a guaranteed £1000 cash or a £2000 prize. To claim yr prize call our customer service representative on 08714712394 between 10am-7pm"}'
```

A blank `text` returns 400. If the joblib file is missing, `/predict` returns 503 and tells you to run `python train.py`.

After training, `pytest` runs the API checks.

## Project structure

```text
train.py            download, split, compare models, cross-validate, print errors, pick threshold, save
top_words.py        tf-idf x coefficient for the top words in a message
api.py              POST /predict
tests/test_api.py   blank text, phish sample, legit sample, response shape
pytest.ini          lets pytest import api.py from the repo root
frontend/           React + Vite page (App.jsx and index.css)
```

## Limitations

The model only knows this SMS collection. It is a bag of words, so order does not matter, and a common word like "call" or "to" can push toward phish even in a normal text. I would not use it as an email filter. Next I would check the threshold on messages from outside this file before adding a more complicated model.

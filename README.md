# PhishCheck

PhishCheck takes a pasted SMS and labels it **phish** or **legit**. It also returns a probability and the words that pushed the score toward phish.

The data file uses `ham` and `spam`. This project shows spam as phish and ham as legit.

## This is SMS spam, not email phishing

Training uses the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection): 5,574 phone texts collected by Almeida, Gómez Hidalgo, and Yamakami. They are old SMS spam (prize draws, premium-rate texts, ordinary chats). They are not modern email phishing. A score here does not tell you whether a current inbox message is an attack.

## Results

`python train.py` dedupes the file, then holds out a stratified 20%. That leaves 5,160 unique texts (4,518 legit, 642 phish). Train is 4,128 messages (514 phish). Test is 1,032 (128 phish). No text appears in both sides.

Baseline, always predict legit (the majority class in training):

- phish precision 0.000
- phish recall 0.000
- confusion matrix (rows actual, columns predicted, order legit then phish): 904 correct legit, 0 false alarms, 128 missed phish, 0 caught

TF-IDF + logistic regression with `class_weight="balanced"`:

- phish precision 0.881
- phish recall 0.930
- confusion matrix: 888 correct legit, 16 false alarms, 9 missed phish, 119 caught

## How it works

`train.py` downloads the UCI file, renames the labels, strips whitespace, and keeps one copy of each text so a repeated SMS cannot sit in both train and test. It then fits a scikit-learn pipeline: `TfidfVectorizer` turns each message into word weights, and `LogisticRegression(class_weight="balanced")` learns a coefficient per word. The saved file is `models/phishcheck_pipeline.joblib`.

For a new message, the API takes one `predict_proba` call. The label is the class with the higher probability. `phish_probability` is the phish entry of that same vector. `top_words` scores each word as its TF-IDF value in this message times its coefficient toward phish, drops blanks and scores of 0 or less, and keeps the top 6. That function is `top_words.py`.

## How to run

Python 3.12 and Node.js. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn api:app --host 127.0.0.1 --port 8741
```

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

Sample buttons on the page fill the box: a prize text and a free-entry text from the collection (phish), "Sorry, I'll call later" from the collection (legit), and a short note about picking up milk (legit).

## Project structure

```text
train.py            download, split, train, print metrics, save the pipeline
top_words.py        tf-idf x coefficient, used by train.py and the API
api.py              POST /predict
tests/test_api.py   blank text, phish sample, legit sample, response shape
pytest.ini          lets pytest import api.py from the repo root
frontend/           React + Vite page (one component, plain CSS)
```

## Limitations

The model only knows this SMS collection. It is a bag of words, so order does not matter, and a common word like "call" or "to" can push toward phish even in a normal text. I would not use it as an email filter. Next I would try the decision threshold on messages from outside this file before adding a more complicated model.

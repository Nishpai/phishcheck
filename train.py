"""Train PhishCheck on the UCI SMS Spam Collection.

Run from the repo root: python train.py
"""

import csv
import json
import urllib.request
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_FILE = Path("data/SMSSpamCollection")
MODEL_PATH = Path("models/phishcheck_pipeline.joblib")
THRESHOLD_PATH = Path("models/threshold.json")


def download_data():
    if DATA_FILE.exists():
        return
    DATA_FILE.parent.mkdir(exist_ok=True)
    zip_path = DATA_FILE.parent / "sms.zip"
    print("Downloading", DATA_URL)
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract("SMSSpamCollection", DATA_FILE.parent)
    zip_path.unlink()


def load_data():
    # Each line is "label<TAB>text". QUOTE_NONE because some texts contain quote marks.
    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    df["text"] = df["text"].str.strip()
    df["label"] = df["label"].map({"spam": "phish", "ham": "legit"})
    print("Loaded messages:", len(df))

    # The file repeats some messages. Keep one copy so the same text
    # can't end up in both train and test.
    df = df.drop_duplicates(subset="text")
    counts = df["label"].value_counts()
    print(f"Unique texts: {len(df)} ({counts['legit']} legit, {counts['phish']} phish)")
    return df


def print_metrics(title, y_true, y_pred):
    precision = precision_score(y_true, y_pred, pos_label="phish", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label="phish", zero_division=0)
    print()
    print(title)
    print(f"phish precision: {precision:.3f}")
    print(f"phish recall:    {recall:.3f}")
    print(f"phish F1:        {f1_score(y_true, y_pred, pos_label='phish', zero_division=0):.3f}")
    print("confusion matrix (rows = actual, columns = predicted, order legit, phish):")
    print(confusion_matrix(y_true, y_pred, labels=["legit", "phish"]))


def main():
    download_data()
    df = load_data()

    # stratify keeps the same phish/legit mix in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"Train: {len(y_train)} ({(y_train == 'phish').sum()} phish)")
    print(f"Test:  {len(y_test)} ({(y_test == 'phish').sum()} phish)")

    # Baseline: always guess the most common label in the training set.
    majority = y_train.value_counts().idxmax()
    print_metrics(f"Baseline (always '{majority}')", y_test, [majority] * len(y_test))

    # class_weight="balanced" because phish is only about 12% of the data,
    # so mistakes on phish need to count more during training.
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("logreg", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    # Naive Bayes with the same TF-IDF settings, to compare against.
    nb_model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb", MultinomialNB()),
    ])

    # 5-fold cross-validation on the training set only. The test set stays unseen.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_train_phish = y_train == "phish"
    print()
    print("5-fold cross-validation on the training set (mean +/- std):")
    for name, pipe in [("Naive Bayes", nb_model), ("Logistic regression", model)]:
        scores = cross_validate(pipe, X_train, y_train_phish, cv=folds,
                                scoring=["precision", "recall", "f1"])
        p, r, f = scores["test_precision"], scores["test_recall"], scores["test_f1"]
        print(f"{name:20} precision {p.mean():.3f} +/- {p.std():.3f}  "
              f"recall {r.mean():.3f} +/- {r.std():.3f}  f1 {f.mean():.3f} +/- {f.std():.3f}")

    nb_model.fit(X_train, y_train)
    print_metrics("TF-IDF + Naive Bayes", y_test, nb_model.predict(X_test))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print_metrics("TF-IDF + logistic regression, threshold 0.5", y_test, y_pred)

    print_errors(X_test, y_test, y_pred)
    threshold = pick_threshold(model, X_train, y_train_phish, folds)

    test_proba = model.predict_proba(X_test)[:, list(model.classes_).index("phish")]
    tuned_pred = ["phish" if p >= threshold else "legit" for p in test_proba]
    print_metrics(f"Test set, threshold {threshold:.3f}", y_test, tuned_pred)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    THRESHOLD_PATH.write_text(json.dumps({"threshold": round(threshold, 4)}))
    print()
    print("Saved", MODEL_PATH, "and", THRESHOLD_PATH)


def print_errors(X_test, y_test, y_pred):
    # Which test messages does the model get wrong?
    false_pos = X_test[(y_test == "legit") & (y_pred == "phish")]
    false_neg = X_test[(y_test == "phish") & (y_pred == "legit")]
    print()
    print(f"False positives (legit marked phish): {len(false_pos)}")
    for text in false_pos:
        print("  ", text[:80])
    print(f"False negatives (phish marked legit): {len(false_neg)}")
    for text in false_neg:
        print("  ", text[:80])


def pick_threshold(model, X_train, y_train_phish, folds):
    # Out-of-fold probabilities: each training message is scored by a model that
    # did not see it, so the test set is never used to pick the cutoff.
    oof = cross_val_predict(model, X_train, y_train_phish, cv=folds, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train_phish, oof)
    # Rule: the threshold with the highest F1, since we care about precision and recall equally.
    # precision/recall have one more entry than thresholds, so drop the last one.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(f1.argmax())
    threshold = float(thresholds[best])
    print()
    print("Out-of-fold precision/recall on the training set near the best F1:")
    print("threshold  precision  recall  f1")
    for i in range(max(best - 3, 0), min(best + 4, len(thresholds))):
        mark = "  <- chosen" if i == best else ""
        print(f"{thresholds[i]:.3f}      {precision[i]:.3f}      {recall[i]:.3f}   {f1[i]:.3f}{mark}")
    return threshold


if __name__ == "__main__":
    main()

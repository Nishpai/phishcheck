#!/usr/bin/env python3
"""Train the PhishCheck classifier on the UCI SMS Spam Collection.

Run from the repo root:

    python train.py

The file labels messages ham or spam. This script calls spam "phish" and ham
"legit", which is the wording PhishCheck will show. The texts are old SMS
spam, not modern email phishing.
"""

import urllib.request
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from top_words import top_phish_words

# Paths sit next to this script, so the run does not depend on the shell's cwd.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
MODEL_PATH = ROOT / "models" / "phishcheck_pipeline.joblib"

# Current UCI download, then the older mirror. The zip holds SMSSpamCollection.
UCI_ZIP_URLS = [
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
]

# 20% of unique texts are held out. random_state keeps the split fixed.
TEST_SIZE = 0.2
RANDOM_STATE = 42


def ensure_dataset() -> Path:
    """Download the UCI zip once, and reuse data/SMSSpamCollection after that."""
    if DATA_FILE.exists():
        print(f"Using existing file: {DATA_FILE.relative_to(ROOT)}")
        return DATA_FILE

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "smsspamcollection.zip"
    downloaded = False
    errors = []

    for url in UCI_ZIP_URLS:
        print(f"Downloading {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "phishcheck-train/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                zip_path.write_bytes(response.read())
            downloaded = True
            break
        except Exception as exc:  # this mirror failed; try the next URL
            errors.append(f"{url} ({exc})")

    if not downloaded:
        details = "\n".join(errors)
        raise SystemExit(f"Could not download the UCI SMS Spam Collection:\n{details}")

    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if Path(name).name == "SMSSpamCollection"]
        if len(members) != 1:
            raise SystemExit(f"Expected one SMSSpamCollection file in the zip, found {members}")
        DATA_FILE.write_bytes(archive.read(members[0]))

    zip_path.unlink()
    print(f"Wrote {DATA_FILE.relative_to(ROOT)}")
    return DATA_FILE


def load_messages(path: Path) -> pd.DataFrame:
    """Read 'label<tab>text' lines. Map spam -> phish and ham -> legit."""
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Older copies of this collection are latin-1.
        content = path.read_text(encoding="latin-1")

    rows = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        if "\t" not in line:
            raise SystemExit(f"Line {line_number} has no tab separator.")

        # Split once: the message itself may contain tabs.
        label, text = line.split("\t", 1)
        # "win now" and "win now " are the same text. Strip before deduping.
        text = text.strip()
        if label == "spam":
            mapped = "phish"
        elif label == "ham":
            mapped = "legit"
        else:
            raise SystemExit(f"Line {line_number}: expected ham or spam, got {label!r}")

        if text:
            rows.append((mapped, text))

    return pd.DataFrame(rows, columns=["label", "text"])


def drop_duplicate_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per text so a repeated SMS cannot land in both splits.

    The collection repeats messages. Texts that carry both labels are dropped.
    """
    loaded = len(df)

    labels_per_text = df.groupby("text")["label"].nunique()
    conflict_texts = labels_per_text[labels_per_text > 1].index
    df = df[~df["text"].isin(conflict_texts)].copy()
    after_conflicts = len(df)

    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    duplicates_removed = after_conflicts - len(df)

    counts = df["label"].value_counts()
    print(f"Loaded messages:           {loaded}")
    print(f"Texts with both labels:    {len(conflict_texts)} (dropped)")
    print(f"Extra copies removed:      {duplicates_removed}")
    print(
        f"Unique texts:              {len(df)} "
        f"({int(counts.get('legit', 0))} legit, {int(counts.get('phish', 0))} phish)"
    )
    return df


def print_phish_metrics(title: str, y_true, y_pred) -> None:
    """Precision and recall for the phish class, then the confusion matrix.

    Precision: of the messages we called phish, how many really were.
    Recall: of the real phish messages, how many we caught.
    Matrix rows are actual labels, columns are predicted labels.
    Order is legit, then phish.
    """
    precision = precision_score(y_true, y_pred, pos_label="phish", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label="phish", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=["legit", "phish"])
    legit_as_legit, legit_as_phish = (int(matrix[0, 0]), int(matrix[0, 1]))
    phish_as_legit, phish_as_phish = (int(matrix[1, 0]), int(matrix[1, 1]))

    print()
    print(title)
    print(f"  phish precision: {precision:.3f}")
    print(f"  phish recall:    {recall:.3f}")
    print("  confusion matrix (row = actual, column = predicted)")
    print(f"  {'':16} {'pred legit':>12} {'pred phish':>12}")
    print(f"  {'actual legit':16} {legit_as_legit:12d} {legit_as_phish:12d}")
    print(f"  {'actual phish':16} {phish_as_legit:12d} {phish_as_phish:12d}")


def main() -> None:
    print("PhishCheck training")
    print("Data: UCI SMS Spam Collection. Spam is labeled phish; ham is labeled legit.")
    print("These are SMS spam messages, not modern email phishing.")
    print()

    df = drop_duplicate_texts(load_messages(ensure_dataset()))
    if df["label"].nunique() < 2:
        raise SystemExit(f"Need both labels to stratify, found {sorted(df['label'].unique())}")

    # Stratified: each side keeps a similar phish/legit mix.
    # Texts are already unique, so nothing is copied across the split.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    overlap = set(X_train) & set(X_test)
    print(f"Train rows:                {len(y_train)} ({int((y_train == 'phish').sum())} phish)")
    print(f"Test rows:                 {len(y_test)} ({int((y_test == 'phish').sum())} phish)")
    print(f"Train/test texts in common: {len(overlap)}")
    if overlap:
        raise SystemExit("The same text is in train and test. Refusing to train on a leaky split.")

    # Baseline uses the training majority only, then is scored on the same test rows.
    majority = y_train.value_counts().idxmax()
    baseline_pred = [majority] * len(y_test)
    print(f"Majority class in training: {majority}")
    print_phish_metrics("Baseline: always predict the majority class", y_test, baseline_pred)

    # TfidfVectorizer: bag of word weights. Rare words count more than common ones.
    # Default tokenizer keeps lowercase words of 2+ characters.
    # class_weight="balanced": phish is rare, so errors on it count more in training.
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    model_pred = pipeline.predict(X_test)
    print_phish_metrics("Model: TF-IDF + balanced logistic regression", y_test, model_pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print()
    print(f"Saved fitted pipeline to {MODEL_PATH.relative_to(ROOT)}")

    # One caught test message, so you can see the same word list the API returns.
    classes = list(pipeline.classes_)
    phish_index = classes.index("phish")
    phish_scores = pipeline.predict_proba(X_test)[:, phish_index]

    best_i = None
    best_score = -1.0
    for i, (actual, predicted, score) in enumerate(zip(y_test, model_pred, phish_scores)):
        if actual == "phish" and predicted == "phish" and score > best_score:
            best_i = i
            best_score = float(score)

    if best_i is None:
        print("\nNo test phish was caught, so there is no sample word list.")
        return

    sample_text = X_test.iloc[best_i]
    shown = sample_text if len(sample_text) <= 240 else sample_text[:240] + "..."
    print()
    print("Sample test message the model caught")
    print("  actual:      phish")
    print("  predicted:   phish")
    print(f"  phish score: {best_score:.3f}")
    print(f"  text:        {shown}")
    print("  words that pushed toward phish (tf-idf x coefficient):")
    pushes = top_phish_words(pipeline, sample_text)
    if not pushes:
        print("    (none)")
        return
    for item in pushes:
        print(f"    {item['word']:<24} {item['contribution']:+.3f}")


if __name__ == "__main__":
    main()

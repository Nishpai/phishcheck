"""Words that pushed one message toward phish.

contribution = tf-idf of the word in this message
               x that word's coefficient for the phish class.

train.py and api.py both call this, so the printed list and the API match.
"""

# Enough words to see the reason, few enough to read in one glance.
TOP_N = 6


def top_phish_words(pipeline, text: str, limit: int = TOP_N) -> list[dict]:
    """Strongest pushes toward phish. Skip blanks and contributions <= 0."""
    vectorizer = pipeline.named_steps["tfidf"]
    logreg = pipeline.named_steps["logreg"]

    # coef_ has one weight per vocabulary word, and it belongs to classes_[1].
    # Flip the sign when that class is legit, so positive always means phish.
    weights = logreg.coef_.ravel()
    if logreg.classes_[1] != "phish":
        weights = -weights

    row = vectorizer.transform([text])
    names = vectorizer.get_feature_names_out()
    scored = []
    for word_id, tfidf_value in zip(row.indices, row.data):
        word = str(names[word_id]).strip()
        if word == "":
            continue
        contribution = float(tfidf_value) * float(weights[word_id])
        if contribution <= 0:
            continue
        scored.append((word, contribution))

    # Highest contribution first. Same contribution → alphabetical word.
    scored.sort(key=lambda item: (-item[1], item[0]))

    top = []
    for word, contribution in scored:
        shown = round(contribution, 3)
        if shown <= 0:
            continue
        top.append({"word": word, "contribution": shown})
        if len(top) == limit:
            break
    return top
